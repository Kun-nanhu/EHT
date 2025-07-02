import prefabricationField
import autograd.numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os


def G_disk():
    phy=prefabricationField.Physics()
    pDe_disk=prefabricationField.PDerivative('disk')
    vel_disk=prefabricationField.Velocity('disk')
    mag=prefabricationField.MagneticField('disk')
    
    ll=2
    n_r0=3
    r_bound=30
    r0s=[0.0+(30-0.001-0.0)*i/(n_r0-1) for i in range(n_r0)]
    disk_results=[]

    for r0 in r0s:
        r_m=max(2.5,r0)
        logr = np.linspace(np.log10(r_m), np.log10(r_bound), ll + 1)
        rs=np.exp(logr*np.log(10))
        for r in rs:
            def integrand(x):
                theta_inj=phy.theta_stag(r0,np.pi/2,x)
                B=mag.B_disk(x,theta_inj)
                return (4.0/3.0)*(pDe_disk.nablaV_disk(x,theta_inj)/(vel_disk.v_disk(x,theta_inj)*B[1]))
            
            integral_value, _ =quad(integrand, r, r_bound)
            G_val=np.exp(integral_value)
            disk_results.append([r,r0,G_val])
    
    disk_results=np.array(disk_results)
    return disk_results

if __name__ == "__main__":
    Gdiskresult=G_disk()
    output_path = os.path.join(os.path.dirname(__file__), 'G_disk.npy')
    np.save(output_path, Gdiskresult)