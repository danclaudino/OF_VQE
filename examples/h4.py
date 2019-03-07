import vqe_methods 
import operator_pools

r = 1.0
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]


#vqe_methods.adapt_vqe(geometry, pool = operator_pools.singlet_SD())
vqe_methods.adapt_vqe(geometry, pool = operator_pools.singlet_GSD())
#vqe_methods.adapt_vqe(geometry, multiplicity =1, reference = 'rohf', ref_state = [0,1,2,3], pool = operator_pools.sf_hs_rohf_GSD())
#vqe_methods.adapt_vqe(geometry, pool = operator_pools.hs_rohf_GSD())
