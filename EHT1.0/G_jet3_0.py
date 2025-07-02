import prefabricationField
import autograd.numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os

def compute_segment_integral(x0, x1, psi, phy, pDe_jet_in, vel_jet, mag):
    def integrand(x):
        r_local = np.exp(x)
        theta = phy.theta_inj_jet(r_local, psi)
        B = mag.B_jet(r_local, theta)
        return np.exp(x) * (4.0 / 3.0) * (
            pDe_jet_in.nablaV_jet(r_local, theta) /
            (vel_jet.v_jet(r_local, theta) * B[1])
        )

    val, _ = quad(integrand, x0, x1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return val

def compute_G_for_psi(psi, r_values, x_values, phy, pDe_jet_in, vel_jet, mag):
    segment_integrals = Parallel(n_jobs=-1)(
        delayed(compute_segment_integral)(x_values[i], x_values[i - 1], psi, phy, pDe_jet_in, vel_jet, mag)
        for i in range(len(x_values) - 1, 0, -1)
    )

    cumulative_integrals = []
    total = 0.0
    for val in segment_integrals:
        total += val
        cumulative_integrals.append(total)

    results = [[r_values[0], psi, 1.0]]

    for r, integral_val in zip(r_values[1:], cumulative_integrals):
        G_val = np.exp(integral_val)
        results.append([r, psi, G_val])

    return results

def G_jet():
    start_time = time.time()

    phy = prefabricationField.Physics()
    pDe_jet_in = prefabricationField.PDerivative('jet', 'in')
    vel_jet = prefabricationField.Velocity('jet')
    mag = prefabricationField.MagneticField('jet')

    ll = 50
    mm = 10
    r_bound = 30

    log_r_values = np.linspace(np.log10(2.5), np.log10(r_bound), ll + 1)
    r_values = 10 ** log_r_values
    x_values = np.log(r_values)
    psis = np.linspace(0, phy.r_H, mm + 1)

    results = Parallel(n_jobs=-1)(
        delayed(compute_G_for_psi)(psi, r_values, x_values, phy, pDe_jet_in, vel_jet, mag)
        for psi in tqdm(psis, desc="Calculating G for each psi")
    )

    jet_results = np.array([item for sublist in results for item in sublist])

    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    return jet_results

if __name__ == "__main__":
    Gjetresult = G_jet()
    output_path = os.path.join(os.path.dirname(__file__), 'G_jet.npy')
    np.save(output_path, Gjetresult)