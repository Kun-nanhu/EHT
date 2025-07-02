import prefabricationField
import autograd.numpy as np
from scipy.integrate import cumulative_trapezoid
import os

def G_jet():
    phy = prefabricationField.Physics()
    pDe_jet_in = prefabricationField.PDerivative('jet', 'in')
    vel_jet = prefabricationField.Velocity('jet')
    mag = prefabricationField.MagneticField('jet')

    ll = 2
    mm = 1
    r_bound = 30
    jet_results = []

    # 生成对数r和r值
    log_r_values = [np.log10(2.5) + (i - 1) * np.log10(r_bound / 2.5) / ll for i in range(1, ll + 2)]
    r_values = [10 ** logr for logr in log_r_values]
    x_values = np.log(r_values)  # x = ln(r)

    psis = np.linspace(0, phy.r_H, mm + 1)

    for psi in psis:
        # 构建积分函数值数组
        integrand_values = []
        for x in x_values:
            r = np.exp(x)
            theta = phy.theta_inj_jet(r, psi)
            B = mag.B_jet(r, theta)
            val = np.exp(x) * (4.0 / 3.0) * (pDe_jet_in.nablaV_jet(r, theta) / (vel_jet.v_jet(r, theta) * B[1]))
            integrand_values.append(val)

        # 使用累积积分，从r_max到r_min反向积分（积分方向影响符号）
        integral_results = cumulative_trapezoid(integrand_values[::-1], x_values[::-1], initial=0)[::-1]

        # 计算G值并保存
        for r, integral in zip(r_values, integral_results):
            G_val = np.exp(integral)
            jet_results.append([r, psi, G_val])

    return np.array(jet_results)

if __name__ == "__main__":
    Gjetresult = G_jet()
    output_path = os.path.join(os.path.dirname(__file__), 'G_jet.npy')
    np.save(output_path, Gjetresult)
