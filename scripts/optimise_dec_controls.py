"""E9: continuum DEC optimisation with radial stress (optimal-control form, ~10 min)."""
import sys, time, math, json, numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chronos import static_spherical as ss
Z = float(sys.argv[1]); rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
x0, K = ss.warm_controls_shell_cluster(Z)
t0 = time.time()
hist = []
for it in range(rounds):
    res = ss.optimise_controls_dec(Z, x0, K, maxiter=500)
    sim = res['sim']
    rec = sim['rec']
    viol_path = float(np.min(rec[:, 3] - np.abs(rec[:, 2])))
    hist.append(dict(it=it, M=res['M_over_rin'], Z=res['Z'], g_end=sim['g_end'], min_path_margin=viol_path,
                     feasible=bool(res['feasible'])))
    print(hist[-1], '%.0fs' % (time.time() - t0), flush=True)
    x0 = res['x']
out = dict(Z=Z, history=hist, x=res['x'].tolist(), h=res['h'].tolist(), rho=res['rho'].tolist(), lam=res['lam'].tolist(),
           rec=rec.tolist())
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'dec_control_opt_Z%g.json' % Z), 'w'))
