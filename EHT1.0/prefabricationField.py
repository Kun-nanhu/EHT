import autograd.numpy as np
from autograd import grad
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import os
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
from scipy.interpolate import interp1d


class Parameters:
    def __init__(self, spin=0.9375):
        #Unit conversion
        self.GeV2cm=1.98e-14 #GeV -> cm
        self.GeV2s=6.58e-25 #GeV -> s
        self.g2GeV=5.62e23 #g -> GeV
        self.B2GeV=1.95e-20 #Gauss -> GeV
        self.pc2cm=3.086e18 # pc -> cm
        self.yr2s=3.15576e7 # yr -> s

        #Physical constants
        self.c=2.99792458e10 #speed of light, (cm/s)
        self.m_planck=1.221e19 #Planck mass=G^-1/2, (GeV)
        self.sigma_T=6.6524586e-25 #Thomson cross section, (cm^2)
        self.m_e=0.511e-3 #electron mass, (GeV)

        #Black Hole parameters
        self.aJ=spin #dimensionless angular momentum value of M87, (1)
        self.M_sun=1.989e33 #mass of the sun, (g)
        self.M_BH=6.5e9 #mass of the black hole, (M_sun)
        self.r_g=(self.M_BH*self.M_sun/self.m_planck**2)*self.g2GeV*self.GeV2cm/self.pc2cm #r_g=GM/c^2, (pc)
        self.r_gE=self.M_BH*self.M_sun*self.g2GeV/self.m_planck**2

class RegionClassifier:
    def __init__(self, a_J=0.9375):
        self.a=a_J
        self.r_H=1+(1-self.a**2)**0.5 #Horizon radius, (r_g)
        self.x=[-0.0011664736219212544-0.000886998813904891j,-0.0011664736219212544+0.000886998813904891j,0.0008537370352110573-0.0012240788722115343j,0.0008537370352110573+0.0012240788722115343j]
        self.r_g=(6.5e9*1.989e33/1.221e19**2)*5.62e23*1.98e-14/3.086e18 #r_g=GM/c^2, (pc)


    def theta_jet(self,r):
        return np.arccos(1-self.r_H/r)
    
    def S(self,r):
        s=0
        for i in range(4):
            s+=(500-200*(self.x[i]/self.r_g)-21*(self.x[i]/self.r_g)**2)/(25+4*(self.x[i]/self.r_g)+3*(self.x[i]/self.r_g)**2+2*(self.x[i]/self.r_g)**3)*np.log(r-(self.x[i]/self.r_g))
        return s

    def theta_stag(self,r,theta,r_stag):
        z=0.5*np.arccos(-(2/r_stag**2)*(2*r_stag+0.5*(-4*r-r**2*np.cos(2*theta)+2*self.S(r))-self.S(r_stag)))
        return z.real

    def theta_disk(self,r):
        return self.theta_stag(0,1,r)

    def classify(self, r, theta):
        theta_j=self.theta_jet(r)
        theta_d=self.theta_disk(r)
        if theta < theta_j:
            return 'jet'
        elif theta > theta_d:
            return 'disk'
        else:
            return 'jet-disk'
        
class Physics:
    '''
    physics field, contains subclass such as MagneticField, VelocityField and NablaV. 
    '''
    def __init__(self, a_J=0.9375):
        self.params=Parameters()
        self.r_g=self.params.r_g
        self.classifier=RegionClassifier(a_J)
        self.aJ=a_J
        self.r_H=1+(1-self.aJ**2)**0.5
        self.Omega_H=self.aJ/self.r_H
        self.x=[-0.0011664736219212544-0.000886998813904891j,-0.0011664736219212544+0.000886998813904891j,0.0008537370352110573-0.0012240788722115343j,0.0008537370352110573+0.0012240788722115343j]

    def Phi(self,r,theta):
        return r*(1-np.cos(theta)) 
    
    def theta_H(self,Phi):
        return np.arccos(1-Phi/self.r_H)
    
    def gphiphi(self,r,theta):
        return (r**2+self.aJ**2+(2*r*self.aJ**2*np.sin(theta)**2)/(r**2+self.aJ**2*np.cos(theta)**2))*np.sin(theta)**2
    
    def Gg(self,Phi):
        return 1+np.cos(self.theta_H(Phi))
    
    def Omega_F(self,Phi):
        if Phi==0:
            return 0
        else:
            sin=np.sin(self.theta_H(Phi))
            lnGg=np.log(self.Gg(Phi))
            Gg=self.Gg(Phi)
            return (sin**2*(1+lnGg)*self.Omega_H)/(4*np.log(2)+sin**2+(sin**2-Gg)*lnGg)
        
    def S(self,r):
        s=0
        for i in range(4):
            s+=(500-200*(self.x[i]/self.r_g)-21*(self.x[i]/self.r_g)**2)/(25+4*(self.x[i]/self.r_g)+3*(self.x[i]/self.r_g)**2+2*(self.x[i]/self.r_g)**3)*np.log(r-(self.x[i]/self.r_g))
        return s

    def theta_stag(self,r,theta,r_stag):
        z=0.5*np.arccos(-(2/r_stag**2)*(2*r_stag+0.5*(-4*r-r**2*np.cos(2*theta)+2*self.S(r))-self.S(r_stag)))
        return z.real
    
    def r_stag_jet(self,Phi):
        if Phi==0:
            return 100
        else:
            return 1/(self.Omega_F(Phi)*np.sqrt(Phi))
    
    def theta_stag_jet(self,Phi):
        return np.arccos(1-Phi/self.r_stag_jet(Phi))
    
    def theta_inj_jet(self,r,Phi):
        return np.arccos(1-Phi/r)

class MagneticField(Physics):
    def __init__(self,name='jet',a_J=0.9375):
        super().__init__(a_J)
        self.aJ=a_J
        self.name=name
        if self.aJ==0.9375:
            self.c=[123,0.679]
        elif self.aJ==0.5:
            self.c=[126,0.345]
        if self.aJ==0.9375:
            self.a=[38.1,0.511,2.34,1.81]
        elif self.aJ==0.5:
            self.a=[19.4,0.395,2.72,1.56]
        if self.aJ==0.9375:
            self.b=[52.1,0.579,2.39,2.29]
        elif self.aJ==0.5:
            self.b=[94.1,0.596,2.70,2.39]
    
    def dPhi(self,Phi):
        return self.c[0]*np.exp(-self.c[1]*Phi)
    
    def B_jet(self,r,theta):
        Phi=self.Phi(r,theta)
        dPhi=self.dPhi(Phi)
        Omega_F=self.Omega_F(Phi)
        gphiphi=self.gphiphi(r,theta)
        B_x=(np.sin(theta)*dPhi)/(2*np.pi*r*(1+np.cos(theta)))
        B_z=dPhi/(2*np.pi*r)
        B_phi=(Omega_F*gphiphi*dPhi)/(2*np.pi*r)
        B=np.sqrt(B_x**2+B_z**2+B_phi**2)
        b_r=(B_x*np.sin(theta)+B_z*np.cos(theta))/B
        b_theta=(-B_z*np.sin(theta)+B_x*np.cos(theta))/B
        return B, b_r, b_theta
    
    def K(self,r):
        return 1000*r**(-4)+100*r**(-3)+8*r**(-2)+4*r**(-1)+1
    
    def B_disk(self,r,theta):
        B_2D=self.b[0]*r**(self.b[1]*np.sin(4*theta-self.b[2])-self.b[3])
        B_phi=self.a[0]*r**(self.a[1]*np.sin(4*theta-self.a[2])-self.a[3])
        B=np.sqrt(B_2D**2+B_phi**2)
        k=self.K(r)
        b_r=(B_2D/B)*((1+k)*np.sin(theta))/np.sqrt(k**2+np.tan(theta)**2)
        if theta==np.pi/2:
            b_theta=-B_2D/B
        else:
            b_theta=-B_2D/B*(np.sin(theta)**2*(1-k/(np.tan(theta)**2)))/(np.cos(theta)*np.sqrt(k**2+np.tan(theta)**2))
        return B, b_r, b_theta

    def B(self,r,theta):
        if self.name=='jet':
            return self.B_jet(r,theta)
        elif self.name=='disk':
            return self.B_disk(r,theta)
        
class Velocity(Physics):
    def __init__(self,name='jet',a_J=0.9375):
        super().__init__(a_J)
        self.name=name

    def V_eff(self,r,Phi):
        return -1/r-self.Omega_F(Phi)**2*(2*Phi*r-Phi**2)/2

    def v_jet(self,r,theta):
        Phi=r*(1-np.cos(theta))
        if theta==0:
            vb=np.sqrt(1-np.exp(2*self.V_eff(r,Phi)))
        else:
            r_stag=self.r_stag_jet(Phi)
            theta_stag=self.theta_stag_jet(Phi)
            Phi_stag=r_stag*(1-np.cos(theta_stag))
            vb=np.sqrt(1-np.exp(2*(self.V_eff(r,Phi)-self.V_eff(r_stag,Phi_stag))))
        if vb<=0.001:
            vb=0.001
        return vb

    def v_disk(self,r,theta):
        return np.sqrt(2/r)

    def v(self,r,theta):
        if self.name=='jet':
            return self.v_jet(r,theta)
        elif self.name=='disk':
            return self.v_disk(r,theta)

class PDerivative(MagneticField,Velocity):
    def __init__(self,name='jet',direction='in',a_J=0.9375):
        super().__init__(name,a_J)
        self.name=name
        self.dir=direction
        self.params=Parameters()
        self.k_theta=(self.params.sigma_T*(1/self.params.GeV2cm**2)*self.params.B2GeV**2)/(6*np.pi*self.params.m_e**2*self.params.r_g)*self.params.r_gE
        self.dr2vr_dr=grad(self.r2vr,0)
        self.dr2sin_dtheta=grad(self.r2sin,0)
        self.dr32br_dr=grad(self.r32br,0)
        self.dsinr12btheta_dtheta=grad(self.sinr12btheta,0)

    def dP_syn(self,B):
        return self.k_theta*self.params.r_g*B**2
    # length unit is in rg

    def r2vr(self,r,theta):
        if theta==0:
            return r**2*self.v_jet(r,theta)
        else:
            return r**2*self.v_jet(r,theta)*self.B_jet(r,theta)[1]
    
    def r2sin(self,theta,r):
        return np.sin(theta)*self.v_jet(r,theta)*self.B_jet(r,theta)[2]
    
    def r32br(self,r,theta):
        return r**1.5*self.B_disk(r,theta)[1]
    
    def sinr12btheta(self,theta,r):
        return np.sin(theta)*r**(-0.5)*self.B_disk(r,theta)[2]

    def nablaV_jet(self,r,theta):
        if theta==0:
            return (1/r**2)*self.dr2vr_dr(r,theta)-self.v_jet(r,theta)/r
        else:
            return (1/r**2)*self.dr2vr_dr(r,theta)+(1/r*np.sin(theta))*self.dr2sin_dtheta(theta,r)

    def nablaV_disk(self,r,theta):
        return np.sqrt(2)*((1/r**2)*self.dr32br_dr(r,theta)+(1/(r*np.sin(theta)))*self.dsinr12btheta_dtheta(theta,r))

    def dP_jet_in(self,r,theta,p):
        B=self.B_jet(r,theta)
        return (-self.dP_syn(B[0])+(p/3)*(self.nablaV_jet(r,theta)))/(self.v_jet(r,theta)*B[1])

    def dP_jet_out(self,r,theta,p):
        B=self.B_jet(r,theta)
        return (self.dP_syn(B[0])+(p/3)*(self.nablaV_jet(r,theta)))/(self.v_jet(r,theta)*B[1])

    def dP_disk(self,r,theta,p):
        B=self.B_disk(r,theta)
        return (-self.dP_syn(B[0])+(p/3)*(self.nablaV_disk(r,theta)))/(self.v_disk(r,theta)*B[1])

    def dP(self,r,theta,p):
        if self.name=='jet':
            if self.dir=='in':
                return self.dP_jet_in(r,theta,p)
            elif self.dir=='out':
                return self.dP_jet_out(r,theta,p)
        elif self.name=='disk':
            return self.dP_disk(r,theta,p)
    
class interpolator():
    def __init__(self):
        self.r0_interpolator = self.r0_inter()

    def r0_inter(self):
        data=np.load(os.path.join(os.path.dirname(__file__), 'r0.npy'))
        rs=data[:,0]
        thetas=data[:,1]
        r0s=data[:,2]

        inter_func=LinearNDInterpolator(list(zip(rs, thetas)), r0s)
        fallback_func=NearestNDInterpolator(list(zip(rs, thetas)), r0s)

        def r0_interr(r,theta):
            value= inter_func(r, theta)
            if np.isnan(value):
                value= fallback_func(r, theta)
            return value
        return r0_interr
    
    def beta_interpolator_jet(self):
        beta_jet_data=np.load(os.path.join(os.path.dirname(__file__),"beta_jet.npy"))
        rss,psiss=np.unique(beta_jet_data[:,0]),np.unique(beta_jet_data[:,1])
        betas=beta_jet_data[:,2].reshape(len(rss),len(psiss))
        betajet_interp=RegularGridInterpolator((rss,psiss),betas,bounds_error=False,fill_value=None)

        def beta_jet(r,theta):
            psi=r*(1-np.cos(theta))
            return betajet_interp((r,psi))
        
        return beta_jet

    
    def beta_interpolator_disk(self):
        disk_data=np.load(os.path.join(os.path.dirname(__file__), 'beta_disk.npy'))
        rs=disk_data[:,0]
        r0s=disk_data[:,1]
        betas=disk_data[:,2]
        
        inter_func=LinearNDInterpolator(list(zip(rs, r0s)), betas)
        fallback_func=NearestNDInterpolator(list(zip(rs, r0s)), betas)
        def beta_disk(r,theta):
            r0=self.r0_interpolator(r,theta)
            value= inter_func((r,r0))
            if np.isnan(value):
                value= fallback_func((r,r0))
            return value
        
        return beta_disk
    
    def xi_interpolator_jet(self):
        jet_data=np.load(os.path.join(os.path.dirname(__file__), 'G_jet.npy'))
        rs,psis=np.unique(jet_data[:,0]),np.unique(jet_data[:,1])
        Gs=jet_data[:,2].reshape(len(rs),len(psis))

        xis=Gs**0.25
        xijet_interp=RegularGridInterpolator((rs,psis),xis,bounds_error=False,fill_value=None)
        def xi_jet(r,theta):
            psi=r*(1-np.cos(theta))
            return xijet_interp((r,psi))
        
        return xi_jet
    
    def xi_interpolator_disk(self):
        disk_data=np.load(os.path.join(os.path.dirname(__file__), 'G_disk.npy'))
        rs=disk_data[:,0]
        r0s=disk_data[:,1]
        Gs=disk_data[:,2]
        
        xis=Gs**0.25
        inter_func=LinearNDInterpolator(list(zip(rs, r0s)), xis)
        fallback_func=NearestNDInterpolator(list(zip(rs, r0s)), xis)
        def xi_disk(r,theta):
            r0=self.r0_interpolator(r,theta)
            value= inter_func((r,r0))
            if np.isnan(value):
                value= fallback_func((r,r0))
            return value
        
        return xi_disk
    
class BH_halo():
    def __init__(self,a_J=0.9375,m_DM=10,sigv=1e-28):
        self.aJ=a_J
        self.m_DM=m_DM
        self.sigv=sigv
        self.parame=Parameters()
        self.rho0=3.3e4
        self.r0=1e5*self.parame.r_g # Sun distance to GC, (pc)
        self.t_BH=1e8 # BH age, (year)
        self.rho_sat=self.m_DM/(self.sigv*self.t_BH*self.parame.yr2s) 
        self.gamma=1
        self.gamma_sp=(9-2*self.gamma)/(4-self.gamma)
        self.r_sat=self.r0*(self.rho0/self.rho_sat)**(1/self.gamma_sp)
        self.r_sp=0.001*(self.parame.M_BH*self.parame.M_sun*self.parame.g2GeV)**(1.5)*(self.rho0)**(-1.5)*(self.r0)**(-3.5)
        self._rmb_interpolator=self._prepare_rmb_grid()

    def rho_sp(self,r):
        return self.rho0*(r/self.r0)**(-self.gamma_sp)
    
    def rho_halo(self,r):
        return self.rho0*(self.r_sp/self.r0)**(-self.gamma_sp)*(r/self.r_sp)**(-1)
    
    def _prepare_rmb_grid(self):
        theta_vals=np.linspace(0, np.pi, 41)
        rmb_vals=[]
        r=sp.Symbol('r',real=True,positive=True)

        for th in theta_vals:
            theta = float(th)
            cos2 = np.cos(theta)**2
            sin = np.sqrt(1 - cos2)
            a = self.aJ

            expr = (r**4 - 4*r**3 - a**2*(1 - 3*cos2)*r**2 + a**4*cos2 +
                    4*a*sin * sp.sqrt(r**5 - a**2 * r**3 * cos2))
            
            try:
                roots=sp.nroots(expr,n=15,maxsteps=100,cleanup=True)
                roots_real=[float(rt.evalf()) for rt in roots if rt.is_real and rt>0]
                if roots_real:
                    rmb_vals.append(max(roots_real))
                else:
                    rmb_vals.append(1.0)
            except:
                rmb_vals.append(1.0)
        
        _rmb_interpolator = RegularGridInterpolator(
            (theta_vals,), np.array(rmb_vals), bounds_error=False, fill_value=None
        )
        return _rmb_interpolator

    def r_crit(self,theta):
        return float(self._rmb_interpolator([theta])[0])*self.parame.r_g
    
    def rho(self,r,theta):
        r_crit=self.r_crit(theta)
        if r<r_crit:
            return 0.0
        elif r<self.r_sat:
            return self.rho_sat
        elif r<self.r_sp:
            return self.rho_sp(r)
        else:
            return self.rho_halo(r)
    
    def plot_equatorial(self, num=200):
        rs = np.logspace(np.log10(0.5*self.r_crit(0)), np.log10(1e6*self.parame.r_g), num)
        rrs=rs/self.parame.r_g
        densities = [self.rho(rval, 0) for rval in rs]
        plt.loglog(rrs, densities)
        plt.xlabel(r'$r\ (\mathrm{Rg})$')
        plt.ylabel(r'$\rho(r,\theta=0)\ (\mathrm{GeV/cm}^3)$')
        plt.title('Dark Matter Spike Density Profile')
        plt.grid(True, which='both', ls='--')
        plt.show()

class DMSpectrum:
    spectab=['e','b','μ','γ','τ']

    def __init__(self,path,mDM,sigv,spec_in):
        self.path=path
        self.mDM=mDM
        self.sigv=sigv
        self.spec=spec_in
        self.dNdEtab,self.norm=self.build_spectrum_table()
        self.interp_func=self.create_interpolator()
        self.dNdE_func=self.make_dNdE()
    
    def build_spectrum_table(self):
        if (self.mDM <= 5) and (self.spec=='e'):
            logm=np.format_float_positional(np.log10(self.mDM),precision=3)
            spectra_name=os.path.join(self.path,'Spectra_e',f'{logm}.txt')
            norm_name=os.path.join(self.path,'Spcetra_e',f'norm_{logm}.txt')
            if not (os.path.exists(spectra_name) and os.path.exists(norm_name)):
                raise FileNotFoundError("Required spectral files not found.")
            dNdEtab=np.loadtxt(spectra_name)
            norm=float(np.loadtxt(norm_name))
        else:
            filename=os.path.join(self.path,'pppc','AtProduction_positrons','AtProduction_positrons.dat')
            if not os.path.exists(filename):
                raise FileNotFoundError("AtProduction_positrons.dat not found.")
            data=np.loadtxt(filename,skiprows=1)
            data=data[np.isclose(data[:,0],self.mDM)]
            if data.size == 0:
                raise ValueError("No data found for the specified DM mass.")
            if self.spec=='e':
                index=4
            elif self.spec=='b':
                index=13
            energies=data[:,1]
            x_vals=10**energies
            E_vals=x_vals*self.mDM
            dNdlogx=data[:,index]
            dNdE_vals=dNdlogx/(E_vals*np.log(10))
            dNdEtab=np.column_stack((E_vals,dNdE_vals))
            norm=1.0
        
        dNdEtab=np.vstack((dNdEtab,[1e10,0.0]))
        return dNdEtab,norm
    
    def create_interpolator(self):
        energies=self.dNdEtab[:,0]
        values =self.dNdEtab[:,1]
        return interp1d(energies,values,kind='linear',bounds_error=False,fill_value=0.0)
    
    def make_dNdE(self):
        def integrand(x):
            return self.interp_func(x)
        
        gaussian_norm,_=quad(integrand,0.7*self.mDM,self.mDM)
        gaussian_norm*=self.norm

        def dNdE(x):
            if not (1e-8*self.mDM<=x<=self.mDM):
                return 0.0
            if self.spec=='e':
                if x<=0.7*self.mDM:
                    return float(self.interp_func(x))
                sigma=0.2*self.mDM
                prefactor=gaussian_norm*2/np.sqrt(2*np.pi*sigma**2)
                return prefactor*np.exp(-(x-self.mDM)**2/(2*sigma**2))
            else:
                return float(self.interp_func(x))
        
        return dNdE
    
    def evaluate(self,x):
        return self.dNdE_func(x)