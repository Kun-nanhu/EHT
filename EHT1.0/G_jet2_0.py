import prefabricationField
import autograd.numpy as np
from scipy.integrate import quad
import os
import time

def G_jet():
    start_time = time.time()

    phy = prefabricationField.Physics()
    pDe_jet_in = prefabricationField.PDerivative('jet', 'in')
    vel_jet = prefabricationField.Velocity('jet')
    mag = prefabricationField.MagneticField('jet')

    ll = 100  # 控制r分段数量（更高密度以提高精度）
    mm = 1    # 控制psi分段数量
    r_bound = 30
    jet_results = []

    # 构造r值（对数间隔）
    log_r_values = np.linspace(np.log10(2.5), np.log10(r_bound), ll + 1)
    r_values = 10 ** log_r_values
    x_values = np.log(r_values)  # x = ln(r)

    psis = np.linspace(0, phy.r_H, mm + 1)

    for psi in psis:
        segment_integrals = []

        # 计算每段积分 r_{i+1} -> r_i（或 x_{i+1} -> x_i）
        for i in range(len(x_values) - 1, 0, -1):
            x0, x1 = x_values[i], x_values[i - 1]

            def integrand(x):
                r_local = np.exp(x)
                theta = phy.theta_inj_jet(r_local, psi)
                B = mag.B_jet(r_local, theta)
                return np.exp(x) * (4.0 / 3.0) * (
                    pDe_jet_in.nablaV_jet(r_local, theta) /
                    (vel_jet.v_jet(r_local, theta) * B[1])
                )

            val, _ = quad(integrand, x0, x1, limit=100, epsabs=1e-9, epsrel=1e-9)
            segment_integrals.append(val)

        # 从r_min开始正向累加积分（计算 r_i 到 r_min 的积分）
        cumulative_integrals = []
        total = 0.0
        for val in segment_integrals:
            total += val
            cumulative_integrals.append(total)

        # 构造r= r_0 的G值 = 1
        jet_results.append([r_values[0], psi, 1.0])

        # 构造其余每个r对应的G值（r_1 到 r_ll）
        for r, integral_val in zip(r_values[1:], cumulative_integrals):
            G_val = np.exp(integral_val)
            jet_results.append([r, psi, G_val])

    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    return np.array(jet_results)

if __name__ == "__main__":
    Gjetresult = G_jet()
    output_path = os.path.join(os.path.dirname(__file__), 'G_jet.npy')
    np.save(output_path, Gjetresult)