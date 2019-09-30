import vqe_methods 
import operator_pools
import numpy as np
import sys 
from joblib import Parallel, delayed

r = 2.0 
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]

#vqe_methods.sgo(geometry, pool=operator_pools.UpCCGSD(), go=False, k=1)
vqe_methods.adapt_vqe(geometry, single_vqe=False, adapt_thresh=1e-10, adapt_maxiter=15, fci_overlap=True, brueckner = 0, fci_nat_orb=1)
#vqe_methods.overlap_adapt_vqe(geometry, adapt_conver='energy')

