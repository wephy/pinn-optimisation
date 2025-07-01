import jax
import jax.numpy as jnp
import optax

class LBFGSWrapper:
    """
    A wrapper for optax.lbfgs to ensure dtype consistency for the linesearch
    state when jax_enable_x64=True. 
    
    It manually casts integer counters in the linesearch state (like 
    'num_linesearch_steps' and 'iterations') to int64 after each update step, 
    preventing a common TypeError with JIT compilation.
    """
    def __init__(self, optimizer_fn, **kwargs):
        """
        Initializes the wrapper.
        
        Args:
            optimizer_fn: The base optimizer function (e.g., optax.lbfgs).
            **kwargs: Arguments to be passed to the base optimizer (e.g., memory_size).
        """
        self.base_optimizer = optimizer_fn(**kwargs)

    def init(self, params):
        """Initializes the state of the base optimizer."""
        return self.base_optimizer.init(params)

    def update(self, updates, state, params=None, **extra_args):
        """
        Performs an update step and then corrects the datatypes in the state.
        """
        # 1. Perform the original update from the base L-BFGS optimizer
        new_updates, new_state = self.base_optimizer.update(updates, state, params, **extra_args)

        # 2. Correct the dtype of the internal state.
        # The full L-BFGS state is a tuple. The last element (-1) is the 
        # state of the linesearch algorithm.
        linesearch_state = new_state[-1]

        # 3. Check if the state has the expected attributes before attempting correction.
        #    Optax's linesearch state is a namedtuple with an 'info' attribute.
        if hasattr(linesearch_state, 'info') and hasattr(linesearch_state.info, 'num_linesearch_steps'):
            
            # Create a new 'info' tuple with the dtypes corrected to int64.
            # We must use ._replace() because namedtuples are immutable.
            new_info = linesearch_state.info._replace(
                num_linesearch_steps=jnp.asarray(linesearch_state.info.num_linesearch_steps, dtype=jnp.int64),
                # It's good practice to also cast the 'iterations' field if it exists.
                # iterations=jnp.asarray(linesearch_state.info.iterations, dtype=jnp.int64)
            )
            
            # Create a new linesearch state with the corrected info
            new_linesearch_state = linesearch_state._replace(info=new_info)
            
            # Reconstruct the full optimizer state tuple with the corrected part
            new_state = new_state[:-1] + (new_linesearch_state,)

        return new_updates, new_state