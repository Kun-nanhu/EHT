import prefabricationField
import autograd.numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os

def compute_segment_integral_disk(r0, x0, x1, phy, pDe_disk, vel_disk, mag):
    def integrand(x):
        theta_inj = phy.theta_stag(r0, np.pi / 2, x)
        B = mag.B_disk(x, theta_inj)
        return (4.0 / 3.0) * (
            pDe_disk.nablaV_disk(x, theta_inj) /
            (vel_disk.v_disk(x, theta_inj) * B[1])
        )

    val, _ = quad(integrand, x0, x1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return val

def compute_G_for_r0(r0, r_values, phy, pDe_disk, vel_disk, mag):
    segment_integrals = Parallel(n_jobs=-1)(
        delayed(compute_segment_integral_disk)(r0, r_values[i], r_values[i+1], phy, pDe_disk, vel_disk, mag)
        for i in range(len(r_values) - 1)
    )

    cumulative_integrals = []
    total = 0.0
    for val in reversed(segment_integrals):
        total += val
        cumulative_integrals.insert(0, total)  

    results = []
    for r, integral_val in zip(r_values, cumulative_integrals):
        G_val = np.exp(integral_val)
        results.append([r, r0, G_val])

    return results

def G_disk():
    start_time = time.time()
    phy = prefabricationField.Physics()
    pDe_disk = prefabricationField.PDerivative('disk')
    vel_disk = prefabricationField.Velocity('disk')
    mag = prefabricationField.MagneticField('disk')

    ll = 50
    n_r0 = 30
    r_bound = 30
    r0s = [0.0 + (30 - 0.001 - 0.0) * i / (n_r0 - 1) for i in range(n_r0)]

    disk_results = []

    for r0 in tqdm(r0s, desc="Calculating G for each r0"):
        r_m = max(2.5, r0)
        log_r_values = np.linspace(np.log10(r_m), np.log10(r_bound), ll + 1)
        r_values = 10 ** log_r_values

        result = compute_G_for_r0(r0, r_values, phy, pDe_disk, vel_disk, mag)
        disk_results.extend(result)

    disk_results = np.array(disk_results)
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    return disk_results

if __name__ == "__main__":
    Gdiskresult = G_disk()
    output_path = os.path.join(os.path.dirname(__file__), 'G_disk.npy')
    np.save(output_path, Gdiskresult)
