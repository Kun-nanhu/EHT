import prefabricationField
import autograd.numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import time

interpolator = prefabricationField.interpolator()
beta_jet_interp = interpolator.beta_interpolator_jet()
xi_jet_interp = interpolator.xi_interpolator_jet()
beta_disk_interp = interpolator.beta_interpolator_disk()
xi_disk_interp = interpolator.xi_interpolator_disk()

def save_file(path, ii, jj, data):
    outfile = Path(path) / "p_inj" / f"r_{ii}" / f"theta_{jj}.npy"
    os.makedirs(outfile.parent, exist_ok=True)
    np.save(outfile, data)

def process_pair(ii, jj, r_vals, theta_vals, cutoff_data, ll, path, classify, phy):
    r_bin = r_vals[ii - 1]
    theta_bin = theta_vals[jj - 1]
    sub = cutoff_data[ii - 1, jj - 1, :, :]
    if sub.size == 0:
        return

    p0_vals = sub[..., 2]
    rcut_vals = sub[..., 3]
    n_pts = len(p0_vals)
    region = classify.classify(r_bin, theta_bin)

    if region == 'jet-disk':
        out = np.zeros((n_pts * ll, 3), dtype=float)
        out[:, 0] = np.repeat(p0_vals, ll)
        save_file(path, ii, jj, out)
        return

    if region == 'jet':
        phi = phy.Phi(r_bin, theta_bin)
        R_stag = phy.r_stag_jet(phi)
        if r_bin > R_stag:
            first = np.linspace(0, 5 / (ll - 26), 30, endpoint=False)
            second = np.linspace(5 / (ll - 26), 1, ll - 30)
            Rinjtab = np.concatenate([first, second])
            rtab = np.vstack([
                rcut_vals[i] + (r_bin - rcut_vals[i]) * Rinjtab for i in range(n_pts)
            ])
            rtab_flat = rtab.ravel()
            theta_tab = np.array([
                phy.theta_inj_jet(rtab_flat[k], phi) for k in range(len(rtab_flat))
            ])
        else:
            first = np.linspace(0, 1, ll - 30, endpoint=False)
            tail_start = (ll - 31) / (ll - 26)
            tail = np.linspace(tail_start, tail_start + 5 / (ll - 26), 30)
            Rinjtab = np.concatenate([first, tail])
            rtab = np.vstack([
                r_bin + (rcut_vals[i] - r_bin) * Rinjtab for i in range(n_pts)
            ])
            rtab_flat = rtab.ravel()
            theta_tab = np.array([
                phy.theta_inj_jet(rtab_flat[k], phi) for k in range(len(rtab_flat))
            ])
    else:
        first = np.linspace(0, 1, ll - 30, endpoint=False)
        tail_start = (ll - 31) / (ll - 26)
        tail = np.linspace(tail_start, tail_start + 5 / (ll - 26), 30)
        Rinjtab = np.concatenate([first, tail])
        rtab = np.vstack([
            r_bin + (rcut_vals[i] - r_bin) * Rinjtab for i in range(n_pts)
        ])
        rtab_flat = rtab.ravel()
        theta_tab = np.array([
            phy.theta_stag(r_bin, theta_bin, rtab_flat[k]) for k in range(len(rtab_flat))
        ])

    pinj = np.empty((n_pts * ll, 3))
    for i in range(n_pts):
        for j in range(ll):
            idx = i * ll + j
            p0 = p0_vals[i]
            r_cur = rtab_flat[idx]
            theta_cur = theta_tab[idx]

            if region == 'jet':
                if r_bin > R_stag:
                    val = ((1 / p0 + beta_jet_interp(r_bin, theta_bin)) *
                           (xi_jet_interp(r_bin, theta_bin) / xi_jet_interp(r_cur, theta_cur))
                           - beta_jet_interp(r_cur, theta_cur)) ** (-1)
                else:
                    val = ((1 / p0 - beta_jet_interp(r_bin, theta_bin)) *
                           (xi_jet_interp(r_bin, theta_bin) / xi_jet_interp(r_cur, theta_cur))
                           + beta_jet_interp(r_cur, theta_cur)) ** (-1)
            else:
                val = ((1 / p0 - beta_disk_interp(r_bin, theta_bin)) *
                       (xi_disk_interp(r_bin, theta_bin) / xi_disk_interp(r_cur, theta_cur))
                       + beta_disk_interp(r_cur, theta_cur)) ** (-1)

            pinj[idx] = (p0, r_cur, val)

    save_file(path, ii, jj, pinj)

def calculatePinj():
    start_time = time.time()
    classify = prefabricationField.RegionClassifier()
    phy = prefabricationField.Physics()

    path = Path.cwd()
    ll = 300
    cutoff_data = np.load(Path(path) / "cutoffPoint.npy")
    r_vals = np.unique(cutoff_data[..., 0])
    theta_vals = np.unique(cutoff_data[..., 1])
    nn = len(r_vals) - 1
    mm = len(theta_vals) - 1

    args_list = [
        (ii, jj, r_vals, theta_vals, cutoff_data, ll, path, classify, phy)
        for ii in range(1, nn + 2)
        for jj in range(1, mm + 2)
    ]

    Parallel(n_jobs=-1)(
        delayed(process_pair)(*args) for args in tqdm(args_list, desc="Computing p_inj", unit="task")
    )

    print(f"Total computation time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    calculatePinj()