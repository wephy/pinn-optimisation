import numpy as np

def wave_1d_time_exact ( c, vxn, vx, vtn, vt ):

  vu = np.zeros ( [ vxn, vtn ] )

  for vti in range ( 0, vtn ):
    for vxi in range ( 0, vxn ):
      vu[vxi,vti] = -np.sin ( np.pi * vx[vxi] ) * np.cos ( np.pi * c * vt[vti] )

  return vu