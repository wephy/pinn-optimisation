import numpy as np

def poisson_1d_exact(vxn, vx):
  u_true = np.zeros(vxn)
  
  for vxi in range(0, vxn):
    u_true[vxi] = vx[vxi] * np.exp(-vx[vxi]**2)
    
  return u_true