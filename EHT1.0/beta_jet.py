import prefabricationField
import autograd.numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os

interpolator = prefabricationField.interpolator()
xi_jet_interp = interpolator.xi_interpolator_jet()

def compute_segment_integral_beta(x0, x1, r, theta, psi, phy, pDe_jet_in, vel_jet, mag):
    def integrand(x):
        r_local = np.exp(x)
        thetax = phy.theta_inj_jet(r_local, psi)
        B = mag.B_jet(r_local, thetax)
        return r_local * (
            pDe_jet_in.dP_syn(B[0]) /
            (vel_jet.v_jet(r_local, thetax) * B[1]) *
            (xi_jet_interp(r_local, thetax) / xi_jet_interp(r, theta))
        )

    val, _ = quad(integrand, x0, x1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return val

def compute_beta_for_psi(psi, r_values, x_values, phy, pDe_jet_in, vel_jet, mag):
    theta_r0 = [phy.theta_inj_jet(r, psi) for r in r_values]
    
    segment_integrals = Parallel(n_jobs=-1)(
        delayed(compute_segment_integral_beta)(
            x_values[i], x_values[i - 1], r_values[i], theta_r0[i], psi,
            phy, pDe_jet_in, vel_jet, mag
        )
        for i in range(len(x_values) - 1, 0, -1)
    )

    cumulative_integrals = []
    total = 0.0
    for val in segment_integrals:
        total += val
        cumulative_integrals.append(total)

    results = [[r_values[0], psi, 0.0]]
    for r, beta_val in zip(r_values[1:], cumulative_integrals):
        results.append([r, psi, beta_val])

    return results

def beta_jet():
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
        delayed(compute_beta_for_psi)(psi, r_values, x_values, phy, pDe_jet_in, vel_jet, mag)
        for psi in tqdm(psis, desc="Calculating beta for each psi")
    )

    beta_results = np.array([item for sublist in results for item in sublist])

    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    return beta_results

if __name__ == "__main__":
    betajetresult = beta_jet()
    output_path = os.path.join(os.path.dirname(__file__), 'beta_jet.npy')
    np.save(output_path, betajetresult)
