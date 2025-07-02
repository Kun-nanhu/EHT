import prefabricationField
import autograd.numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import time
import os

def compute_point(i, j, k, r_grid, theta_grid, p_list, R_bound, p_inj_down,
                  phy, pDe_jet_in, pDe_jet_out, pDe_disk, classify):

    r = r_grid[i]
    theta = theta_grid[j]
    p = p_list[k]
    r_start = r
    r_end = R_bound
    region = classify.classify(r, theta)

    if region == 'jet':
        Phi = phy.Phi(r, theta)
        r_stag = phy.r_stag_jet(Phi)
        if r < r_stag:
            r_end = min(R_bound, r_stag)
            def ode_fun(R, P): return pDe_jet_in.dP(R, phy.theta_inj_jet(R, Phi), P)
        else:
            r_end = r_stag
            def ode_fun(R, P): return pDe_jet_out.dP(R, phy.theta_inj_jet(R, Phi), P)
    elif region == 'disk':
        def ode_fun(R, P): return pDe_disk.dP(R, phy.theta_stag(r, theta, R), P)
    else:  # 'jet-disk'
        return [r, theta, p, 0.0]

    def hit_aim(R, P): return P - p_inj_down
    hit_aim.terminal = True
    hit_aim.direction = -1

    direction = np.sign(r_end - r_start)
    if direction == 0:
        return [r, theta, p, r_end]

    sol = solve_ivp(
        fun=ode_fun,
        t_span=(r_start, r_end),
        y0=[1 / p],
        events=hit_aim,
        max_step=abs(r_end - r_start) / 100,
    )

    if len(sol.t_events[0]) > 0:
        return [r, theta, p, sol.t_events[0][0]]
    else:
        return [r, theta, p, r_end]

def cutoffPoint():
    start_time = time.time()
    phy = prefabricationField.Physics()
    pDe_jet_in = prefabricationField.PDerivative('jet', 'in')
    pDe_jet_out = prefabricationField.PDerivative('jet', 'out')
    pDe_disk = prefabricationField.PDerivative('disk')
    classify = prefabricationField.RegionClassifier()

    R_min = 2.5
    R_max = 2.5 * 12
    R_bound = 30
    mm = 9  
    nn = 9  
    n_ps=8
    p_inj_down = np.float64(1e-6)

    logr_grid = np.linspace(np.log10(R_min), np.log10(R_max), nn + 1)
    r_grid = np.exp(logr_grid * np.log(10))
    theta_grid = np.linspace(0, 0.5 * np.pi - 1e-3, mm + 1)
    p_list = np.logspace(-3, 4, n_ps)

    param_combinations = list(product(range(nn+1), range(mm+1), range(len(p_list))))

    results_list = Parallel(n_jobs=-1)(
        delayed(compute_point)(i, j, k, r_grid, theta_grid, p_list, R_bound, p_inj_down,
                               phy, pDe_jet_in, pDe_jet_out, pDe_disk, classify)
        for (i, j, k) in tqdm(param_combinations)
    )

    results = np.zeros((nn+1, mm+1, len(p_list), 4), dtype=float)
    for idx, (i, j, k) in enumerate(param_combinations):
        results[i, j, k] = results_list[idx]

    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    return results

if __name__ == "__main__":
    cutoffresults = cutoffPoint()
    output_path = os.path.join(os.path.dirname(__file__), 'cutoffPoint.npy')
    np.save(output_path, cutoffresults)
