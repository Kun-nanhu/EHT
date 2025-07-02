import prefabricationField
import autograd.numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os

interpolator = prefabricationField.interpolator()
xi_disk_interp = interpolator.xi_interpolator_disk()

def compute_segment_integral(x0, x1, r_ref, r0, theta_ref, phy, pDe_disk, vel_disk, mag):
    def integrand(x):
        theta = phy.theta_stag(r0, np.pi / 2, x)
        B = mag.B_disk(x, theta)
        return (pDe_disk.dP_syn(B[0]) / (vel_disk.v_disk(x, theta) * B[1])) * (xi_disk_interp(x, theta) / xi_disk_interp(r_ref, theta_ref))

    val, _ = quad(integrand, x0, x1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return val

def compute_beta_for_r0(r0, r_values, phy, pDe_disk, vel_disk, mag):
    segment_integrals = []
    theta_ref_list = [phy.theta_stag(r0, np.pi / 2, r) for r in r_values]

    x_values = r_values  # Since integration variable is x=r
    for i in range(len(x_values) - 1):
        x0, x1 = x_values[i], x_values[i+1]
        segment_integrals.append((x0, x1, r_values[i], theta_ref_list[i]))

    results = []

    # Parallel segment integration
    segment_vals = Parallel(n_jobs=-1)(
        delayed(compute_segment_integral)(x0, x1, r_ref, r0, theta_ref, phy, pDe_disk, vel_disk, mag)
        for (x0, x1, r_ref, theta_ref) in segment_integrals
    )

    total = 0.0
    results.append([r_values[-1], r0, 0.0])  # Terminal beta is 0 at r_max

    for i in reversed(range(len(r_values) - 1)):
        total += segment_vals[i]
        results.append([r_values[i], r0, total])

    results.sort(key=lambda x: x[0])  # Ensure results are in increasing r order
    return results

def beta_disk():
    start_time = time.time()

    phy = prefabricationField.Physics()
    pDe_disk = prefabricationField.PDerivative('disk', 'in')
    vel_disk = prefabricationField.Velocity('disk')
    mag = prefabricationField.MagneticField('disk')

    ll = 50
    n_r0 = 30
    r_bound = 30
    r0s = [0.0 + (30 - 0.001 - 0.0) * i / (n_r0 - 1) for i in range(n_r0)]

    results = []
    for r0 in tqdm(r0s, desc="Processing r0"):
        r_m = max(2.5, r0)
        log_r_values = np.linspace(np.log10(r_m), np.log10(r_bound), ll + 1)
        r_values = 10 ** log_r_values
        results += compute_beta_for_r0(r0, r_values, phy, pDe_disk, vel_disk, mag)

    beta_results = np.array(results)
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")

    return beta_results

if __name__ == "__main__":
    betadiskresult = beta_disk()
    output_path = os.path.join(os.path.dirname(__file__), 'beta_disk.npy')
    np.save(output_path, betadiskresult)
