import sys, math
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos import static_spherical as ss
for K in [1,2,3,4,6,8,12]:
    r=ss.optimise_nested_shells(10.0,K,n_starts=25,seed=1)
    if r: 
        sh=[(round(s,4),round(p/s,4)) for s,p in r['shells']]
        print(K, round(r['M_over_rin'],5), 'Z=',round(r['Z'],4), 'max p/sigma=',max(p/s for s,p in r['shells']))
    else: print(K,None)
