import prefabricationField
import autograd.numpy as np
import os
from pathlib import Path
from tqdm import tqdm

interpolator=prefabricationField.interpolator()
beta_jet_interp = interpolator.beta_interpolator_jet()
xi_jet_interp = interpolator.xi_interpolator_jet()
beta_disk_interp = interpolator.beta_interpolator_disk()
xi_disk_interp = interpolator.xi_interpolator_disk()

def save_fe(path, ii, jj, data):
    outfile = Path(path)/"f_e" /f"r_{ii}"/f"theta_{jj}.npy"
    os.makedirs(outfile.parent, exist_ok=True)
    np.save(outfile, data)

def calculatefe(mDM,sigv,spec):
    classify = prefabricationField.RegionClassifier()
    parame=prefabricationField.Parameters()
    phy = prefabricationField.Physics()
    vel=prefabricationField.Velocity()
    rho=prefabricationField.BH_halo(a_J=0.9375,m_DM=mDM,sigv=sigv).rho
    path=str(Path.cwd())
    dm_spectrum=prefabricationField.DMSpectrum(path,mDM,sigv,spec)

    path = Path.cwd()
    cutoff_data=np.load(path/"cutoffPoint.npy")
    p_inj_folder=os.path.join(path,"p_inj" if mDM>1 else "p_inj_subGeV")
    r_vals = np.unique(cutoff_data[...,0])
    theta_vals = np.unique(cutoff_data[...,1])
    p0_vals = np.unique(cutoff_data[...,2])
    nn = len(r_vals)-1
    mm = len(theta_vals)-1
    ll=len(p0_vals)

    total_points = (mm+1)*(nn+1)
    progress=tqdm(total=total_points, desc="Computing DM integral")
    for ii,r in enumerate(r_vals,start=1):
        for jj, theta in enumerate(theta_vals,start=1):
            region= classify.classify(r, theta)
            if region=='jet-disk':
                out=np.zeros((ll,2),dtype=float)
                out[:,0]=np.repeat(p0_vals,ll)
                save_fe(path, ii, jj, out)
                progress.update(1)
                continue
            if region=='jet':
                phi=phy.Phi(r,theta)
                thetaInj=phy.theta_inj_jet
                vR=vel.v_jet
                xi=xi_jet_interp
            else:
                thetaInj=phy.theta_stag
                vR=vel.v_disk
                xi=xi_disk_interp

            p_inj_folder=Path(p_inj_folder)
            target_path=p_inj_folder/ f"r_{ii}"/ f"theta_{jj}.npy"
            pinjtab= np.load(target_path)
            nrows=pinjtab.shape[0]
            nblocks=nrows//ll
            
            fe=np.zeros((ll,2),dtype=float)
            for vari in range(ll):
                p0=pinjtab[vari*nblocks,0]
                total=0.0
                for m in range(ll-1):
                    idx=vari*ll+m
                    r_inj=pinjtab[idx,1]
                    dr=abs(r_inj-pinjtab[idx+1,1])
                    p_inj=pinjtab[idx,2]
                    if region=='jet':
                        theta_inj=thetaInj(r_inj,phi)
                    else:
                        theta_inj=thetaInj(r,theta,r_inj)
                    vR_inj=vR(r_inj,theta_inj)

                    term=(parame.pc2cm/(parame.c*8*np.pi))*((sigv*rho(r_inj,theta_inj)**2)/(p_inj**2*mDM**2))*dm_spectrum.evaluate(p_inj)
                    term*=(1/vR_inj)*(p_inj/p0)**4*(xi(r,theta)/xi(r_inj,theta_inj))**4
                    term*=dr
                fe[vari]=(p0,total)
            save_fe(path, ii, jj, fe)
            progress.update(1)
    progress.close()
    return None

if __name__=="__main__":
    mDM=10
    sigv=1e-28
    spec='e'
    calculatefe(mDM,sigv,spec)