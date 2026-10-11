"""E20: nested thin shells under DEC plus per-shell causal radial stability (~10 min)."""
import sys, time, json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chronos import static_spherical as ss
out = {}
for b2max in [1.0, 1/3]:
    for Z in [2.0, 5.0, 10.0]:
        best = None
        for K in [1, 2, 3, 4, 6, 8, 12, 16]:
            t0 = time.time()
            r = ss.optimise_nested_shells(Z, K, n_starts=30, beta2_max=b2max)
            if r and (best is None or r['M_over_rin'] < best['M_over_rin']):
                best = dict(r, K=K)
            print(f'beta2max={b2max:.3f} Z={Z} K={K}: ' + (f"M/r_in={r['M_over_rin']:.4f}" if r else 'infeasible') + f'  ({time.time()-t0:.0f}s)', flush=True)
        if best:
            out[f'{b2max:.3f}_{Z}'] = dict(M_over_rin=best['M_over_rin'], K=best['K'], R=best['R'].tolist(), m=best['m'].tolist())
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'stable_nested_shells.json'), 'w'), indent=1)
print('done', flush=True)
