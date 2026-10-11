import sys, math
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos import static_spherical as ss
r=ss.optimise_nested_shells(10.0,8,n_starts=25,seed=1)
for (s,p),R,m in zip(r['shells'],r['R'],r['m']):
    print(f'R={R:.4f} y_out={2*m/R:.4f} sigma={s:.3e} p={p:.3e} p-sigma={p-s:.2e}')
