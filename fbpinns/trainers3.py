"""
Defines trainer classes for FBPINNs and PINNs.

This is the main entry point for training FBPINNs and PINNs.

To train a FBPINN / PINN, use a Constants object to set up the problem and define its hyperparameters, and pass it
to one of the trainer classes defined here
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, jvp
from jax import random
import optax
import numpy as np

from fbpinns.trainers_base import _Trainer
from fbpinns import networks, plot_trainer
from fbpinns.util.logger import logger
from fbpinns.util.jax_util import tree_index, total_size, str_tensor, partition, combine, flops_cost_analysis


# LABELLING CONVENTIONS

# xd = dimensionality of point
# ud = dimensionality of solution
# dims = (ud, xd)
# n = number of points
# m = number of models (i.e. subdomains)
# c = number of constraints

# x = single coordinate (xd)
# x_batch = batch of coordinates (n, xd)
# uj = solution and gradient component list

# j = index in uj
# im = index of model
# ip = index of point
# ic = index of constraint
# i = generic index

# nm = shape of rectangular DDs
# ii = for grid index in nm


# STATIC FUNCTIONS

def tree_map_dicts(f, *trees):
    "Apply function f to corresponding dictionaries in pytrees."

    def is_dict_leaf(x):
        return isinstance(x, dict)

    def map_fn(*sub_trees):
        if is_dict_leaf(sub_trees[0]):
            return f(*sub_trees)
        return sub_trees[-1]

    return jax.tree_util.tree_map(map_fn, *trees, is_leaf=is_dict_leaf)


def get_jmaps(required_ujs):
    "Generate tree for computing chained jacobians"
    tree = {}
    for iu,ixs in required_ujs:
        t = tree
        for ix in ixs:
            if ix not in t:
                t[ix] = {}
            t = t[ix]
    def get_nodes(t, n=(), ks=()):
        ni = len(n)-1 + 1
        for k in t:
            ks_ = ks+(k,)
            if t[k]:
                n += (((ni,k),ks_,0),)
                n = get_nodes(t[k], n, ks_)
            else:
                n += (((ni,k),ks_,1),)
        return n
    nodes = get_nodes(tree)
    leaves = tuple((i + 1, node[1]) for i,node in enumerate(nodes) if node[2])
    if not leaves:
        leaves = ((0,()),)
    jac_is = ()
    for iu,ixs in required_ujs:
        io = len(ixs)
        il = [leaf[1][:io] for leaf in leaves].index(ixs)
        jac_is += ((il, io, iu),)
    return nodes, leaves, jac_is



# JITTED FUNCTIONS

def FBPINN_model_inner(params, x, norm_fn, network_fn, unnorm_fn, window_fn):
    x_norm = norm_fn(params, x)
    u_raw = network_fn(params, x_norm)
    u = unnorm_fn(params, u_raw)
    w = window_fn(params, x)
    return u*w, w, u_raw

def PINN_model_inner(all_params, x, norm_fn, network_fn, unnorm_fn):
    x_norm = norm_fn(all_params, x)
    u_raw = network_fn(all_params, x_norm)
    u = unnorm_fn(u_raw)
    return u, u_raw

def FBPINN_model(all_params, x_batch, takes, model_fns, verbose=True):
    "Defines FBPINN model"
    norm_fn, network_fn, unnorm_fn, window_fn, constraining_fn = model_fns
    m_take, n_take, p_take, np_take, npou = takes
    x_take = x_batch[n_take]
    log_ = logger.info if verbose else logger.debug
    log_("x_batch"); log_(str_tensor(x_batch))
    log_("x_take"); log_(str_tensor(x_take))
    d = all_params
    all_params_take = {t_k: {cl_k: {k: jax.tree_util.tree_map(lambda p:p[m_take], d[t_k][cl_k][k]) if k=="subdomain" else d[t_k][cl_k][k]
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    f = {t_k: {cl_k: {k: jax.tree_util.tree_map(lambda p: 0, d[t_k][cl_k][k]) if k=="subdomain" else jax.tree_util.tree_map(lambda p: None, d[t_k][cl_k][k])
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f,0,None,None,None,None))(all_params_take, x_take, norm_fn, network_fn, unnorm_fn, window_fn)
    u = jnp.concatenate([us, ws], axis=1)
    u = jax.ops.segment_sum(u, p_take, indices_are_sorted=False, num_segments=len(np_take))
    wp = u[:,-1:]
    u = u[:,:-1]/wp
    u = jax.ops.segment_sum(u, np_take, indices_are_sorted=False, num_segments=len(x_batch))
    u = u/npou
    u = constraining_fn(all_params, x_batch, u)
    return u, wp, us, ws, us_raw

def PINN_model(all_params, x_batch, model_fns, verbose=True):
    "Defines PINN model"
    norm_fn, network_fn, unnorm_fn, constraining_fn = model_fns
    u, u_raw = vmap(PINN_model_inner, in_axes=(None,0,None,None,None))(all_params, x_batch, norm_fn, network_fn, unnorm_fn)
    u = constraining_fn(all_params, x_batch, u)
    return u, u_raw

def FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps):
    "Computes gradients of FBPINN model"
    def u(x_batch):
        return FBPINN_model(all_params, x_batch, takes, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def PINN_forward(all_params, x_batch, model_fns, jmaps):
    "Computes gradients of PINN model"
    def u(x_batch):
        return PINN_model(all_params, x_batch, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def _get_ujs(x_batch, jmaps, u):
    nodes, leaves, jac_is = jmaps
    vs = jnp.tile(jnp.eye(x_batch.shape[1]), (x_batch.shape[0],1,1))
    fs = [u]
    for (ni, ix), _, _ in nodes:
        fs.append(jacfwd(fs[ni], vs[:,ix]))
    jacs = []
    for ie,_ in leaves:
        fin, jac = fs[ie](x_batch)
        jacs.append(jac+(fin,))
    ujs = [jacs[il][io][:,iu:iu+1] for il,io,iu in jac_is]
    return ujs

def jacfwd(f, v):
    "Computes jacobian for single x, for all y, fully chained"
    def jacfun(x):
        y, j, aux = jvp(f, (x,), (v,), has_aux=True)
        aux = aux + (y,)
        return j, aux
    return jacfun

def FBPINN_loss(active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn):
    d, da = active_params, fixed_params
    trainable_params = {cl_k: {k: jax.tree_util.tree_map(lambda p1, p2:jnp.concatenate([p1,p2],0), d[cl_k][k], da[cl_k][k]) if k=="subdomain" else d[cl_k][k]
        for k in d[cl_k]}
        for cl_k in d}
    all_params = {"static":static_params, "trainable":trainable_params}
    constraints_ = []
    for takes, jmaps, constraint in zip(takess, jmapss, constraints):
        x_batch = constraint[0]
        ujs = FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps)
        constraints_.append(constraint+ujs)
    return loss_fn(all_params, constraints_)

def PINN_loss(active_params, static_params, constraints, model_fns, jmapss, loss_fn):
    all_params = {"static":static_params, "trainable":active_params}
    constraints_ = []
    for jmaps, constraint in zip(jmapss, constraints):
        x_batch = constraint[0]
        ujs = PINN_forward(all_params, x_batch, model_fns, jmaps)
        constraints_.append(constraint+ujs)
    return loss_fn(all_params, constraints_)

@partial(jit, static_argnums=(0, 5, 8, 9, 10))
def FBPINN_update(optimiser_fn, active_opt_states,
                  active_params, fixed_params, static_params_dynamic, static_params_static,
                  takess, constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)

    # Define a function that computes the loss for a given set of parameters.
    # This is needed for the L-BFGS line search.
    def _value_fn(p):
        return FBPINN_loss(p, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn)

    # update step
    lossval, grads = value_and_grad(_value_fn, argnums=0)(active_params)
    
    # Pass all required arguments to the optimiser for LBFGS compatibility
    updates, active_opt_states = optimiser_fn(
        grads, active_opt_states, active_params,
        value=lossval,
        grad=grads,
        value_fn=_value_fn
    )
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params

@partial(jit, static_argnums=(0, 4, 6, 7, 8))
def PINN_update(optimiser_fn, active_opt_states,
                active_params, static_params_dynamic, static_params_static,
                constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)

    # Define a function that computes the loss for a given set of parameters.
    # This is needed for the L-BFGS line search.
    def _value_fn(p):
        return PINN_loss(p, static_params, constraints, model_fns, jmapss, loss_fn)

    # update step
    lossval, grads = value_and_grad(_value_fn, argnums=0)(active_params)

    # Pass all required arguments to the optimiser for LBFGS compatibility
    updates, active_opt_states = optimiser_fn(
        grads, active_opt_states, active_params,
        value=lossval,
        grad=grads,
        value_fn=_value_fn
    )
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params

# For fast test inference only

@partial(jax.jit, static_argnums=(1,4,5))
def _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return FBPINN_model(all_params, x_batch, takes, model_fns, verbose)
def FBPINN_model_jit(all_params, x_batch, takes, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose)

@partial(jax.jit, static_argnums=(1,3,4))
def _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return PINN_model(all_params, x_batch, model_fns, verbose)
def PINN_model_jit(all_params, x_batch, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose)


def get_inputs(x_batch, active, all_params, decomposition):
    "Get the inputs to the FBPINN model based on x_batch and the active models"
    n_take, m_take, training_ims = decomposition.inside_points(all_params, x_batch)
    active = jnp.array(active).copy()
    assert jnp.isin(active, jnp.array([0,1,2])).all()
    assert active.shape == (all_params["static"]["decomposition"]["m"],)
    active = active.at[active==0].set(1)
    mask = jnp.zeros_like(active)
    mask = mask.at[training_ims].set(1)
    active = active*mask
    ims_ = jnp.arange(all_params["static"]["decomposition"]["m"])
    active_ims = ims_[active==1]
    fixed_ims = ims_[active==2]
    all_ims = jnp.concatenate([active_ims, fixed_ims])
    inv = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    inv = inv.at[all_ims].set(jnp.arange(len(all_ims)))
    m_take = inv[m_take]
    pous = all_params["static"]["decomposition"]["subdomain"]["pou"][all_ims].astype(int)
    np_ = jnp.stack([n_take, pous[m_take,0]], axis=-1).astype(int)
    npu,p_take = jnp.unique(np_, axis=0, return_inverse=True)
    p_take = p_take.reshape(-1)
    np_take = npu[:,0]
    npou = len(jnp.unique(all_params["static"]["decomposition"]["subdomain"]["pou"].astype(int)))
    takes = (m_take, n_take, p_take, np_take, npou)
    def cut_active(d):
        return {cl_k: {k: jax.tree_util.tree_map(lambda p:p[active_ims], d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_fixed(d):
        return {cl_k: {k: jax.tree_util.tree_map(lambda p:p[fixed_ims],  d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_all(d):
        return {cl_k: {k: jax.tree_util.tree_map(lambda p:p[all_ims],    d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def merge_active(da, d):
        for cl_k in d:
            for k in d[cl_k]:
                if k=="subdomain":
                    d[cl_k][k] = jax.tree_util.tree_map(lambda pa, p: p.copy().at[active_ims].set(pa), da[cl_k][k], d[cl_k][k])
                else:
                    d[cl_k][k] = da[cl_k][k]
        return d
    return takes, all_ims, (active, cut_active, cut_fixed, cut_all, merge_active)


def _common_train_initialisation(c, key, all_params, problem, domain):
    # print stats
    logger.info("Total number of trainable parameters:")
    for k in all_params["trainable"]:
        logger.info(f'\t{k}: {total_size(all_params["trainable"][k]):,}')

    key, subkey = random.split(key)
    constraints_global = problem.sample_constraints(all_params=all_params, domain=domain, key=subkey, sampler=c.sampler, batch_shapes=c.ns)
    for constraint_ in constraints_global:
        for c_ in constraint_[:-1]:
            assert c_.shape[0] == constraint_[0].shape[0]
    x_batch_global = jnp.concatenate([constraint_[0] for constraint_ in constraints_global])# (n, xd)
    constraint_offsets_global = jnp.array([0]+[constraint_[0].shape[0] for constraint_ in constraints_global[:-1]], dtype=int).cumsum()# (c,) offset index of each constraint
    constraint_fs_global = jnp.zeros((x_batch_global.shape[0], len(constraints_global)), dtype=bool)# (n, c)
    for ic in range(len(constraints_global)):# fill in constraint filters
        constraint_fs_global = constraint_fs_global.at[
            constraint_offsets_global[ic]:constraint_offsets_global[ic]+constraints_global[ic][0].shape[0], ic].set(True)
    required_ujss = [constraint_[-1] for constraint_ in constraints_global]
    constraints_global = [constraint_[:-1] for constraint_ in constraints_global]
    
    x_batch_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    logger.info("Computing exact solution..")
    u_exact = problem.exact_solution(all_params=all_params, x_batch=x_batch_test, batch_shape=c.n_test)
    logger.info("Computing done")
    
    
    loss_fn = problem.loss_fn
    jmapss = tuple(get_jmaps(required_ujs) for required_ujs in required_ujss)
    
    return (loss_fn, key,
            constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
            x_batch_test, u_exact)

class FBPINNTrainer(_Trainer):
    "FBPINN model trainer class"

    def _get_current_stage(self, i):
        """Helper to find the optimiser stage index for a given step i."""
        schedule = self.c.optimiser_schedule
        step_boundaries = np.cumsum([s[1] for s in schedule])
        stage_idx = np.searchsorted(step_boundaries, i, side='right')
        return stage_idx

    def _get_x_batch(self, i, active, all_params, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition):
        "Get the x_batch points from x_batch_global which are inside active models"
        ims = jnp.arange(all_params["static"]["decomposition"]["m"])[active==1]
        training_ips, _d = decomposition.inside_models(all_params, x_batch_global, ims)
        x_batch = x_batch_global[training_ips]
        logger.info(f"[i: {i}/{self.c.n_steps}] Average number of points/dimension in active subdomains: {_d:.2f}")
        constraint_fs = constraint_fs_global[training_ips]
        ix_ = jnp.arange(x_batch.shape[0])
        constraint_ips = [ix_[f] for f in constraint_fs.T]
        constraints = [[c_[training_ips[constraint_ips[ic]]-constraint_offsets_global[ic]]
                        for c_ in constraints_global[ic]]
                       for ic in range(len(constraints_global))]
        return x_batch, constraints, constraint_fs, constraint_ips

    def _get_update_inputs(self, i, active, all_params, all_opt_states, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition, problem):
        "Get inputs to the FBPINN update step based on active models"
        start0 = time.time()
        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs..")
        x_batch, constraints, constraint_fs, constraint_ips = self._get_x_batch(i, active, all_params, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition)
        takes, _, (active, cut_active, cut_fixed, cut_all, merge_active) = get_inputs(x_batch, active, all_params, decomposition)
        active_params = cut_active(all_params["trainable"])
        fixed_params = cut_fixed(all_params["trainable"])
        static_params = cut_all(all_params["static"])
        def find_and_cut_dicts(tree_all):
            def do_cut(sub_tree_all):
                if isinstance(sub_tree_all, dict):
                    return cut_active(sub_tree_all)
                return sub_tree_all
            return jax.tree_util.tree_map(do_cut, tree_all, is_leaf=lambda x: isinstance(x, dict))
        active_opt_states = find_and_cut_dicts(all_opt_states)
        (m_take, n_take, p_take, np_take, npou) = takes
        takess = []
        iu_ = jnp.arange(np_take.shape[0])
        for f, ips in zip(constraint_fs.T, constraint_ips):
            f1 = f
            f2 = f[np_take]
            ius = iu_[f2]
            f3 = f1[n_take]
            f4 = f2[p_take]
            inv = jnp.zeros(x_batch.shape[0], dtype=int)
            inv = inv.at[ips].set(jnp.arange(len(ips)))
            inv2 = jnp.zeros(np_take.shape[0], dtype=int)
            inv2 = inv2.at[ius].set(jnp.arange(len(ius)))
            m_take_ic = m_take[f3]
            n_take_ic = inv[n_take[f3]]
            p_take_ic = inv2[p_take[f4]]
            np_take_ic = inv[np_take[f2]]
            takess.append((m_take_ic, n_take_ic, p_take_ic, np_take_ic, npou))
        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs done ({time.time()-start0:.2f} s)")
        return active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch

    def train(self):
        "Train model"
        c, writer = self.c, self.writer
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)
        all_params = {"static":{},"trainable":{}}
        domain, problem, decomposition = c.domain, c.problem, c.decomposition
        for tag, cl, kwargs in zip(["domain", "problem", "decomposition"], [domain, problem, decomposition],
                                   [c.domain_init_kwargs, c.problem_init_kwargs, c.decomposition_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] == all_params["static"]["problem"]["dims"][1] == all_params["static"]["decomposition"]["xd"])
        logger.info(f'Total number of subdomains: {all_params["static"]["decomposition"]["m"]}')
        network = c.network
        key, *subkeys = random.split(key, all_params["static"]["decomposition"]["m"]+1)
        args_ = c.network_init_kwargs.values()
        ps_ = vmap(network.init_params, in_axes=(0,)+(None,)*len(args_))(jnp.array(subkeys), *args_)
        if ps_[0]: all_params["static"]["network"] = {"subdomain": ps_[0]}
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}
        model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn, decomposition.window_fn, problem.constraining_fn)
        scheduler = c.scheduler(all_params=all_params, n_steps=c.n_steps, **c.scheduler_kwargs)
        (loss_fn, key, constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
        x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)

        logger.info("Getting test data inputs..")
        active_test_ = jnp.ones(all_params["static"]["decomposition"]["m"], dtype=int)
        takes_, all_ims_, (_, _, _, cut_all_, _)  = get_inputs(x_batch_test, active_test_, all_params, decomposition)
        test_inputs = (takes_, all_ims_, cut_all_)
        schedule = c.optimiser_schedule
        step_boundaries = np.cumsum([s[1] for s in schedule])
        current_stage = -1
        optimiser_fn = None
        all_opt_states = None
        pstep, fstep, u_test_losses, train_loss_history, test_error_history = 0, 0, [], [], []
        start0, start1, report_time = time.time(), time.time(), 0.
        merge_active, active_params, active_opt_states, fixed_params = None, None, None, None
        lossval = None
        
        step_keys = random.split(key, c.n_steps)
        
        for i, active_ in enumerate(scheduler):
            
            subkey = step_keys[i]
            
            stage_idx = np.searchsorted(step_boundaries, i, side='right')
            if stage_idx > current_stage:
                current_stage = stage_idx
                optimiser_class, n_stage_steps, optimiser_kwargs = schedule[stage_idx]
                if i != 0 and merge_active:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                logger.info(f"Starting stage {stage_idx} with {optimiser_class.__name__} for {n_stage_steps} steps at global step {i}")
                print("\n" + "="*80)
                print(f"VERIFICATION AT STEP {i}: OPTIMISER HAS BEEN SWITCHED TO: {optimiser_class.__name__}")
                print(f"The new optimiser object is: {optimiser_class}")
                print("="*80 + "\n")
                optimiser = optimiser_class(**optimiser_kwargs)
                all_opt_states = optimiser.init(all_params["trainable"])
                optimiser_fn = optimiser.update
                
            if active_ is not None:
                active = active_
                
                if i != 0:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                    all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)
                    
                # then get new inputs to update step
                active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch = \
                     self._get_update_inputs(i, active, all_params, all_opt_states, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition, problem)

                startc = time.time()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling update step..")
                static_params_dynamic, static_params_static = partition(static_params)
                update = FBPINN_update.lower(optimiser_fn, active_opt_states,
                                             active_params, fixed_params, static_params_dynamic, static_params_static,
                                             takess, constraints, model_fns, jmapss, loss_fn).compile()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
                p,f = total_size(active_params["network"]), flops_cost_analysis(update.cost_analysis())
                logger.debug(f"p, f: {p}, {f}")
            
            if i == 0:
                self._report(i, pstep, fstep, train_loss_history, test_error_history, start0, start1, report_time,
                            u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                            active, merge_active, active_opt_states, active_params, x_batch,
                            lossval)
            lossval, active_opt_states, active_params = update(active_opt_states,
                                         active_params, fixed_params, static_params_dynamic,
                                         takess, constraints)
            pstep, fstep = pstep+p, fstep+f
            if lossval is not None:
                train_loss_history.append((i + 1, lossval.item(), schedule[current_stage][0].__name__))
            start1, report_time = self._report(i + 1, pstep, fstep, train_loss_history, test_error_history, start0, start1, report_time,
                        u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                        active, merge_active, active_opt_states, active_params, x_batch_global,
                        lossval)
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")
        if merge_active:
            all_params["trainable"] = merge_active(active_params, all_params["trainable"])
        return all_params

    def _report(self, i, pstep, fstep, train_loss_history, test_error_history, start0, start1, report_time,
                u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                active, merge_active, active_opt_states, active_params, x_batch,
                lossval):
        "Report results"
        c = self.c
        summary_,test_,model_save_ = [(i % f == 0) for f in [c.summary_freq, c.test_freq, c.model_save_freq]]
        if summary_ or test_ or model_save_:
            if i != 0 and summary_:
                rate = c.summary_freq / (time.time()-start1-report_time)
                self._print_summary(i, lossval.item(), rate, start0)
                start1, report_time = time.time(), 0.
            if test_ or model_save_:
                start2 = time.time()
                if merge_active:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                if test_:
                    self._test(
                        x_batch_test, u_exact, train_loss_history, test_error_history, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition)
                if model_save_:
                    self._save_model(i, (i, all_params, all_opt_states, active, jnp.array(test_error_history)))
                report_time += time.time()-start2
        return start1, report_time

    def _test(self, x_batch_test, u_exact, train_loss_history, test_error_history, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition):
        "Test step"
        c, writer = self.c, self.writer
        takes, all_ims, cut_all = test_inputs
        all_params_cut = {"static":cut_all(all_params["static"]), "trainable":cut_all(all_params["trainable"])}
        u_test, wp_test_, us_test_, ws_test_, us_raw_test_ = FBPINN_model_jit(all_params_cut, x_batch_test, takes, model_fns, verbose=False)
        if all_params["static"]["problem"]["dims"][1] == 1:
            m, ud, n = all_params["static"]["decomposition"]["m"], all_params["static"]["problem"]["dims"][0], x_batch_test.shape[0]
            us_test, ws_test, us_raw_test = jnp.full((m, n, ud), jnp.nan), jnp.full((m, n, 1), jnp.nan), jnp.full((m, n, ud), jnp.nan)
            us_test = us_test.at[all_ims[takes[0]], takes[1], :].set(us_test_)
            ws_test = ws_test.at[all_ims[takes[0]], takes[1], :].set(ws_test_)
            us_raw_test = us_raw_test.at[all_ims[takes[0]], takes[1], :].set(us_raw_test_)
            us_test = us_test.at[all_ims[takes[0]], takes[1], :].divide(wp_test_[takes[2]])/takes[4]
            ws_test = ws_test.at[all_ims[takes[0]], takes[1], :].divide(wp_test_[takes[2]])/takes[4]
            us_test = vmap(model_fns[-1], in_axes=(None,None,0))(all_params, x_batch_test, us_test)
        else:
            us_test, ws_test, us_raw_test = us_test_, ws_test_, us_raw_test_
        l2_rel_err = jnp.linalg.norm(u_exact - u_test) / jnp.linalg.norm(u_exact)
        if i > 0 : # Don't record error for step 0 as it's before first update
             test_error_history.append((i, l2_rel_err.item(), self.c.optimiser_schedule[self._get_current_stage(i-1)][0].__name__))
        writer.add_scalar("loss/test/l2_rel_err_istep", l2_rel_err, i)
        if i % (c.test_freq * 5) == 0:
            fs = plot_trainer.plot("FBPINN", all_params["static"]["problem"]["dims"],
                x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, c.n_test, train_loss_history=train_loss_history, test_error_history=test_error_history)
            if fs is not None:
                self._save_figs(i, fs)
        return None # u_test_losses is no longer used

class PINNTrainer(_Trainer):
    "PINN model trainer class"

    def _get_current_stage(self, i):
        """Helper to find the optimiser stage index for a given step i."""
        schedule = self.c.optimiser_schedule
        step_boundaries = np.cumsum([s[1] for s in schedule])
        stage_idx = np.searchsorted(step_boundaries, i, side='right')
        return stage_idx
        
    def train(self):
        "Train model"
        c, writer = self.c, self.writer
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)
        all_params = {"static":{},"trainable":{}}
        domain, problem = c.domain, c.problem
        for tag, cl, kwargs in zip(["domain", "problem"], [domain, problem],
                                   [c.domain_init_kwargs, c.problem_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] == all_params["static"]["problem"]["dims"][1])
        network = c.network
        key, subkey = random.split(key)
        ps_ = network.init_params(key=subkey, **c.network_init_kwargs)
        if ps_[0]: all_params["static"]["network"] = {"subdomain": ps_[0]}
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}
        mu_, sd_ = c.decomposition_init_kwargs["unnorm"]
        unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
        model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)
        (loss_fn, key, training_batch_data, jmapss,
        x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)
        # if not c.resample_every_step:
        
        constraints_all, _, _, _ = training_batch_data
            
        schedule = c.optimiser_schedule
        step_boundaries = np.cumsum([s[1] for s in schedule])
        current_stage = -1
        optimiser_fn, all_opt_states = None, None
        active_params = all_params["trainable"]
        static_params = all_params["static"]
        active_opt_states, update = None, None
        pstep, fstep, train_loss_history, test_error_history = 0, 0, [], []
        start0, start1, report_time = time.time(), time.time(), 0.
        lossval = None
        step_keys = random.split(key, c.n_steps)
        constraints = None
        for i in range(c.n_steps):
            # if c.resample_every_step:
            #     if i % c.resample_frequency == 0:
            #         logger.debug(f"Resampling new collocation points at step {i}...")
            #         if c.sampler != "uniform":
            #             # logger.warning(f"On-the-fly generation is slow with '{c.sampler}' sampler. 'uniform' is recommended.")
            #             pass
            #         subkey = step_keys[i]
            #         constraints = [c[:-1] for c in problem.sample_constraints(all_params=all_params, domain=domain, key=subkey, sampler=c.sampler, batch_shapes=c.ns)]
            # else:
            # n_batches = c.n_batches if c.n_batches > 0 else 1
            subkey = step_keys[i]
            # batch_idx = random.randint(subkey, (), 0, n_batches)
            constraints = constraints_all[batch_idx]
                
            stage_idx = np.searchsorted(step_boundaries, i, side='right')
            if stage_idx > current_stage:
                current_stage = stage_idx
                optimiser_class, n_stage_steps, optimiser_kwargs = schedule[stage_idx]
                if i != 0:
                    all_params["trainable"] = active_params
                logger.info(f"Starting stage {stage_idx} with {optimiser_class.__name__} for {n_stage_steps} steps at global step {i}")
                optimiser = optimiser_class(**optimiser_kwargs)
                active_params = all_params["trainable"]
                active_opt_states = optimiser.init(active_params)
                optimiser_fn = optimiser.update
                startc = time.time()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling update step for new stage..")
                static_params_dynamic, static_params_static = partition(static_params)
                update = PINN_update.lower(optimiser_fn, active_opt_states,
                                           active_params, static_params_dynamic, static_params_static,
                                           constraints, model_fns, jmapss, loss_fn).compile()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
                p,f = total_size(active_params["network"]), flops_cost_analysis(update.cost_analysis())
                logger.debug(f"p, f: {p}, {f}")
            if i == 0:
                x_batch = constraints[0][0]
                start1, report_time = self._report(i, pstep, fstep, train_loss_history, test_error_history, start0, start1, report_time,
                            u_exact, x_batch_test, all_params, active_opt_states, model_fns, problem,
                            active_opt_states, active_params,
                            x_batch, 
                            lossval)
            lossval, active_opt_states, active_params = update(active_opt_states,
                                       active_params, static_params_dynamic,
                                       constraints)
            pstep, fstep = pstep+p, fstep+f
            if lossval is not None:
                train_loss_history.append((i + 1, lossval.item(), schedule[current_stage][0].__name__))
            x_batch = constraints[0][0]
            start1, report_time = self._report(i + 1, pstep, fstep, train_loss_history, test_error_history, start0, start1, report_time,
                        u_exact, x_batch_test, all_params, active_opt_states, model_fns, problem,
                        active_opt_states, active_params,
                        x_batch,
                        lossval)
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")
        all_params["trainable"] = active_params
        return all_params

    def _report(self, i, pstep, fstep, train_loss_history, test_error_history, start0, start1, report_time,
                u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                active_opt_states, active_params,
                x_batch,
                lossval):
        "Report results"
        c = self.c
        summary_,test_,model_save_ = [(i % f == 0) for f in [c.summary_freq, c.test_freq, c.model_save_freq]]
        if summary_ or test_ or model_save_:
            if i != 0 and summary_:
                rate = c.summary_freq / (time.time()-start1-report_time)
                self._print_summary(i, lossval.item(), rate, start0)
                start1, report_time = time.time(), 0.
            if test_ or model_save_:
                start2 = time.time()
                all_params["trainable"] = active_params
                all_opt_states = active_opt_states
                if test_:
                    self._test(
                        x_batch_test, u_exact, train_loss_history, test_error_history, x_batch, i, pstep, fstep, start0, all_params, model_fns, problem)
                if model_save_:
                    self._save_model(i, (i, all_params, all_opt_states, jnp.array(test_error_history)))
                report_time += time.time()-start2
        return start1, report_time

    def _test(self, x_batch_test, u_exact, train_loss_history, test_error_history, x_batch, i, pstep, fstep, start0, all_params, model_fns, problem):
        "Test step"
        c, writer = self.c, self.writer
        u_test, u_raw_test = PINN_model_jit(all_params, x_batch_test, model_fns, verbose=False)
        l2_rel_err = jnp.linalg.norm(u_exact-u_test) / jnp.linalg.norm(u_exact)
        if i > 0: # Don't record error for step 0
            test_error_history.append((i, l2_rel_err.item(), self.c.optimiser_schedule[self._get_current_stage(i-1)][0].__name__))
        writer.add_scalar("loss/test/l2_rel_err_istep", l2_rel_err, i)
        if i % (c.test_freq * 5) == 0:
            fs = plot_trainer.plot("PINN", all_params["static"]["problem"]["dims"],
                x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, c.n_test, train_loss_history=train_loss_history, test_error_history=test_error_history)
            if fs is not None:
                self._save_figs(i, fs)
        return None



if __name__ == "__main__":

    from fbpinns.constants import Constants
    from fbpinns.problems import HarmonicOscillator1D, HarmonicOscillator1DHardBC, HarmonicOscillator1DInverse

    logger.setLevel("DEBUG")

    c = Constants(
        run="test",
        #problem=HarmonicOscillator1D,
        #problem=HarmonicOscillator1DHardBC,
        problem=HarmonicOscillator1DInverse,
        network_init_kwargs = dict(layer_sizes=[1, 32, 32, 1]),
        # resample_every_step=True, # Enable on-the-fly generation
        sampler="uniform" # Make sure to use a fast sampler
        )

    run = FBPINNTrainer(c)
    #run = PINNTrainer(c)

    all_params = run.train()
    print(all_params["static"]["problem"])
    if "problem" in all_params["trainable"]:
        print(all_params["trainable"]["problem"])