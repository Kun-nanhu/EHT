import prefabricationField
import autograd.numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def cutoffPoint():
    phy=prefabricationField.Physics()
    pDe_jet_in=prefabricationField.PDerivative('jet','in')
    pDe_jet_out=prefabricationField.PDerivative('jet','out')
    pDe_disk=prefabricationField.PDerivative('disk')
    classify=prefabricationField.RegionClassifier()

    # grid parameters
    R_min=2.5 
    R_max=2.5*12
    R_bound=30
    mm=1 # for theta
    nn=1 # for r
    p_inj_down=np.float64(1e-6)

    logr_grid = np.linspace(np.log10(R_min), np.log10(R_max), nn + 1)
    r_grid = np.exp(logr_grid*np.log(10))
    theta_grid = np.linspace(0, 0.5*np.pi - 1e-3, mm + 1)
    p_list=np.logspace(-3,4,2)

    def cutoffPoint(r_grid,theta_grid):
        results=np.zeros((nn+1,mm+1,len(p_list),4),dtype=float)
        for i, r in enumerate(r_grid):
            for j, theta in enumerate(theta_grid):
                r_start=r
                r_end=R_bound
                # judge the region
                region=classify.classify(r,theta)
                if region=='jet':
                    Phi=phy.Phi(r,theta)
                    r_stag=phy.r_stag_jet(Phi)
                    if r<r_stag:
                        r_end=min(R_bound,r_stag)
                        ode_fun = lambda R, P: pDe_jet_in.dP(R,phy.theta_inj_jet(R,Phi), P)
                    else:
                        r_end=r_stag
                        ode_fun = lambda R, P: pDe_jet_out.dP(R,phy.theta_inj_jet(R,Phi), P)
                elif region=='disk':
                    ode_fun = lambda R, P: pDe_disk.dP(R,phy.theta_stag(r,theta,R), P)

                for k, p in enumerate(p_list):
                    results[i,j,k,0]=r
                    results[i,j,k,1]=theta
                    results[i,j,k,2]=p
                    if region=='jet-disk':
                        results[i,j,k,3]=0.0
                    else:
                        def hit_aim(R,P): return P-p_inj_down
                        hit_aim.terminal = True
                        hit_aim.direction = -1

                        direction=np.sign(r_end-r_start)
                        if direction==0:
                            results[i,j,k,3]=r_end
                            continue
                        sol=solve_ivp(
                            fun=ode_fun,
                            t_span=(r_start, r_end),
                            y0=[1/p],
                            events=hit_aim,
                            max_step=abs(r_end-r_start)/100,
                        )
                        evts=sol.t_events[0]
                        if len(evts)>0:
                            results[i,j,k,3]=sol.t_events[0][0]
                        else:
                            results[i,j,k,3]=r_end
        return results
    
    results=cutoffPoint(r_grid,theta_grid)
    return results

cutoffresults=cutoffPoint()
np.save('cutoffPoint.npy',cutoffresults)

datacutoff=np.load("cutoffPoint.npy")
print(datacutoff.shape)
for i in range(datacutoff.shape[0]):
    for j in range(datacutoff.shape[1]):
        for k in range(datacutoff.shape[2]):
            print(datacutoff[i,j,k,0],datacutoff[i,j,k,1],datacutoff[i,j,k,2],datacutoff[i,j,k,3])