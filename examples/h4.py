import vqe_methods 
import operator_pools
import sys 
import random

r = 2.5
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]

vqe_methods.ucc(geometry, pool = operator_pools.UpCCGSD(), ops_order = [i for i in range(12)], k=1)
