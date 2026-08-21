"""
Supplementary figure: fast-slow dissection of the impaired (IMP) neuron.
Two panels:
  (a) fast-subsystem bifurcation diagram in (z, v) with the slow nullclines
      v = Z0 - z overlaid for several Z0 (shows Z0 sliding the operating
      point across the fixed saddle-node fold);
  (b) fast-subsystem firing rate f(z) (continuous onset at z* -> Class-I / SNIC).

Run:
    python supp_figure_fast_slow.py AdEx_models/IMP_cr_Z0_-50.json
"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit


# ---------------- parameters (repo JSON -> SI units) ----------------------
jp = sys.argv[1] if len(sys.argv) > 1 else 'CMEI-Epilepsy/AdEx_models/IMP_cr_Z0_-50.json'
m = json.load(open(jp))[0][0]['model']
gl=m['g_L']*1e-6; El=m['E_L']*1e-3; Dt=m['Delta_T']*1e-3; Vt=m['V_th']*1e-3
a=m['a']*1e-9; b=m['b']*1e-9; gp=m['gp']*1e-6; bz=m['bz']
Cm=m['C_m']*1e-9; tauw=m['tau_w']*1e-3; Vr=m['V_reset']*1e-3
Vcut=m['V_peak_detect']*1e-3; tref=m['t_ref']*1e-3
Z0_paper=m['Z0']*1e-3
mV=1e3

# ---------------- fast-subsystem fixed points + stability -----------------
def F(v, z):
    th=Vt-bz*z
    return -(gl+a)*(v-(El+z))+gl*Dt*np.exp(np.clip((v-th)/Dt,-50,50))-gp*z

def roots(z):
    vg=np.linspace(-0.090,-0.030,60001); Fv=F(vg,z); s=np.sign(Fv); out=[]
    for i in np.where(np.diff(s)!=0)[0]:
        vf=vg[i]-Fv[i]*(vg[i+1]-vg[i])/(Fv[i+1]-Fv[i]); th=Vt-bz*z
        J11=(-gl+gl*np.exp(np.clip((vf-th)/Dt,-50,50)))/Cm
        eig=np.linalg.eigvals([[J11,-1/Cm],[a/tauw,-1/tauw]])
        out.append((vf, bool(np.all(eig.real<0))))
    return out

# ---------------- fast-subsystem firing rate f(z) (numba) -----------------
@njit(cache=True)
def _fz(z, T, dt, settle):
    v=Vr; w=0.0; ref=0.0; n=0; ns=int(T/dt); ks=int(settle/dt)
    for k in range(ns):
        if ref>0.0:
            ref-=dt; v=Vr
        else:
            th=Vt-bz*z; arg=(v-th)/Dt
            if arg>20.0: arg=20.0
            dv1=(-gl*(v-(El+z))+gl*Dt*np.exp(arg)-w-gp*z)/Cm
            dw1=(a*(v-(El+z))-w)/tauw
            vp=v+dt*dv1
            if vp>Vcut:
                v=Vr; w+=b; ref=tref
                if k>ks: n+=1
            else:
                arg2=(vp-th)/Dt
                if arg2>20.0: arg2=20.0
                dv2=(-gl*(vp-(El+z))+gl*Dt*np.exp(arg2)-(w+dt*dw1)-gp*z)/Cm
                dw2=(a*(vp-(El+z))-(w+dt*dw1))/tauw
                v=v+0.5*dt*(dv1+dv2); w=w+0.5*dt*(dw1+dw2)
                if v>Vcut:
                    v=Vr; w+=b; ref=tref
                    if k>ks: n+=1
    return n/((ns-ks)*dt)

# ---------------- compute branches, f(z), equilibria ----------------------
zg=np.linspace(-0.020,0.0138,400)
zs=[];vs=[];zu=[];vu=[]
for z in zg:
    for vf,st in roots(z):
        (zs.append(z) or vs.append(vf)) if st else (zu.append(z) or vu.append(vf))
zs,vs,zu,vu=map(np.array,(zs,vs,zu,vu))

order=np.argsort(zs); zso,vso=zs[order],vs[order]; g=vso+zso   # v_rest(z)+z = Z0 at eq.
Z0_crit=g.max()
z_star=zso[np.argmax(g)]
def z_eq(Z0): return None if Z0>Z0_crit else np.interp(Z0,g,zso)

z_scan=np.linspace(-0.005,0.045,111)
fz=np.array([_fz(z,4.0,2e-5,1.0) for z in z_scan])

# ---------------- plot ----------------------------------------------------
fig,(axA,axB)=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={'width_ratios':[1.45,1]})

# Panel (a)
axA.axvspan(z_star*mV,25,color='#cb181d',alpha=0.06)
axA.text(19.5,-43,'no stable rest\n→ firing',color='#cb181d',fontsize=8.5,ha='center')
axA.plot(zs*mV,vs*mV,'.',ms=3,color='#238b45',label='stable rest branch')
axA.plot(zu*mV,vu*mV,'.',ms=3,color='#cb181d',label='unstable threshold branch')
axA.axvline(z_star*mV,color='0.5',ls=':',lw=1.3)
axA.text(z_star*mV+0.3,-70.6,'fold  z*',color='0.4',fontsize=9)
Z0_list=[-0.070,-0.060,Z0_paper,Z0_crit,-0.045]
cmap=plt.cm.viridis(np.linspace(0.12,0.85,len(Z0_list)))
zz=np.linspace(-0.020,0.025,50)
for Z0,c in zip(Z0_list,cmap):
    tag=f'$Z_0$={Z0*mV:.1f} mV'
    if abs(Z0-Z0_crit)<1e-6: tag+=' (critical)'
    if abs(Z0-Z0_paper)<1e-6: tag+=' — paper'
    axA.plot(zz*mV,(Z0-zz)*mV,'-',lw=1.6,color=c,label=tag)
    ze=z_eq(Z0)
    if ze is not None:
        axA.plot(ze*mV,(Z0-ze)*mV,'o',ms=10,color=c,mec='k',mew=0.8,zorder=6)
axA.annotate('rising $Z_0$ slides the\nequilibrium toward the fold',
             xy=(11,-63.5),xytext=(-8,-47),fontsize=8.5,
             arrowprops=dict(arrowstyle='->',color='0.3'))
axA.set_xlabel('slow variable  $z$  (mV)'); axA.set_ylabel('$v$  (mV)')
axA.set_xlim(-20,25); axA.set_ylim(-72,-40)
axA.legend(fontsize=7.5,loc='lower left'); axA.set_title('(a)',loc='left',fontweight='bold')

# Panel (b)
axB.plot(z_scan*mV,fz,color='#2171b5',lw=2)
axB.axvline(z_star*mV,color='0.5',ls=':',lw=1.3)
axB.text(z_star*mV+0.6,axB.get_ylim()[1]*0.06,'onset $z^{*}$\n(rate → 0)',fontsize=8)
axB.set_xlabel('slow variable  $z$  (mV)')
axB.set_ylabel('fast-subsystem firing rate  (Hz)')
axB.set_title('(b)',loc='left',fontweight='bold')

fig.tight_layout()
fig.savefig('supp_fast_slow.png', dpi=200, bbox_inches='tight')
fig.savefig('supp_fast_slow.svg', bbox_inches='tight')
fig.savefig('supp_fast_slow.pdf', bbox_inches='tight')
print(f'z*={z_star*mV:.2f} mV | Z0_crit={Z0_crit*mV:.2f} mV | Z0_paper={Z0_paper*mV:.0f} -> z_eq={z_eq(Z0_paper)*mV:.2f} mV')
print('saved supp_fast_slow.png / .pdf')
