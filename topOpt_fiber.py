import os
import glob
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from PIL import Image

# Configuration for output resolution
FIG_WIDTH_IN, FIG_HEIGHT_IN = 6, 3    # inches
DPI = 300                           # dpi for PNG
GIF_DURATION_MS = 200               # ms per frame

# Material properties for transverse isotropy (fiber aligned along local 1-axis)
E1, E2 = 1.0, 0.5      # axial and transverse moduli
nu12 = 0.3             # major Poisson's ratio
G12 = 0.2              # in-plane shear

# Optimization parameters
alpha_theta = 0.1      # step size for theta update
move_theta = np.pi/18  # max change per iteration (~10 deg)


def constitutive_C(theta):
    nu21 = nu12 * E2 / E1
    denom = 1 - nu12 * nu21
    C_loc = np.array([
        [E1/denom, nu12*E2/denom, 0],
        [nu12*E2/denom, E2/denom, 0],
        [0, 0, G12]
    ])
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([
        [ c*c, s*s, 2*s*c],
        [ s*s, c*c, -2*s*c],
        [-s*c, s*c, c*c - s*s]
    ])
    return T.T @ C_loc @ T


def KE_aniso(theta):
    C = constitutive_C(theta)
    gp = [(-1/np.sqrt(3),-1/np.sqrt(3)), (1/np.sqrt(3),-1/np.sqrt(3)),
          (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
    coords = np.array([[0,0],[1,0],[1,1],[0,1]])
    KE = np.zeros((8,8))
    for xi, eta in gp:
        dN = 0.25 * np.array([[-(1-eta),(1-eta),(1+eta),-(1+eta)],
                               [-(1-xi),-(1+xi),(1+xi),(1-xi)]])
        J = dN @ coords
        detJ = np.linalg.det(J)
        B = np.zeros((3,8))
        dN_xy = np.linalg.inv(J) @ dN
        for i in range(4):
            B[:,2*i:2*i+2] = [[dN_xy[0,i],0],[0,dN_xy[1,i]],[dN_xy[1,i],dN_xy[0,i]]]
        KE += B.T @ C @ B * detJ
    return KE


def FE(nelx, nely, x, theta, penal):
    ndof = 2*(nely+1)*(nelx+1)
    K = lil_matrix((ndof, ndof))
    F = np.zeros((ndof,1))
    U = np.zeros((ndof,1))
    for i in range(nelx):
        for j in range(nely):
            ke = KE_aniso(theta[j,i])
            n1 = j + i*(nely+1)
            n2 = j + (i+1)*(nely+1)
            ed = np.array([2*n1,2*n1+1,2*n2,2*n2+1,2*n2+2,2*n2+3,2*n1+2,2*n1+3])
            K[np.ix_(ed,ed)] += x[j,i]**penal * ke
    F[1,0] = -1.0
    fixed = np.concatenate((np.arange(0,2*(nely+1),2),[ndof-1]))
    free = np.setdiff1d(np.arange(ndof), fixed)
    U[free,0] = spsolve(K[free,:][:,free],F[free,0])
    return U


def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros_like(x)
    rminf = int(np.floor(rmin))
    for i in range(nelx):
        for j in range(nely):
            wsum = 0; val = 0
            for ii in range(max(i-rminf,0),min(i+rminf+1,nelx)):
                for jj in range(max(j-rminf,0),min(j+rminf+1,nely)):
                    w = rmin - np.hypot(i-ii,j-jj)
                    if w>0:
                        wsum += w
                        val += w * x[jj,ii] * dc[jj,ii]
            dcn[j,i] = val/(x[j,i]*wsum) if wsum>0 else dc[j,i]
    return dcn


def OC(nelx, nely, x, volfrac, dc):
    l1, l2 = 0, 1e5; move=0.2
    while l2-l1>1e-4:
        mid=(l1+l2)/2
        xnew = np.maximum(0.001, np.minimum(1, np.maximum(x-move, np.minimum(x+move, x*np.sqrt(-dc/mid)))))
        if xnew.sum()>volfrac*nelx*nely: l1=mid
        else: l2=mid
    return xnew


def filter_theta(theta, rmin):
    cf, sf = np.cos(theta), np.sin(theta)
    cff = np.zeros_like(cf); sff = np.zeros_like(sf)
    rminf = int(np.floor(rmin))
    for i in range(theta.shape[1]):
        for j in range(theta.shape[0]):
            wsum=csum=ssum=0
            for ii in range(max(i-rminf,0),min(i+rminf+1,theta.shape[1])):
                for jj in range(max(j-rminf,0),min(j+rminf+1,theta.shape[0])):
                    w=rmin-np.hypot(i-ii,j-jj)
                    if w>0: wsum+=w; csum+=w*cf[jj,ii]; ssum+=w*sf[jj,ii]
            cff[j,i]=csum/wsum; sff[j,i]=ssum/wsum
    return np.arctan2(sff,cff)


def top(nelx,nely,volfrac,penal,rmin,iters=50):
    os.makedirs('results',exist_ok=True)
    x = volfrac*np.ones((nely,nelx))
    theta = np.zeros((nely,nelx))
    compliance_hist = []
    # initial frame
    fig,ax=plt.subplots(figsize=(FIG_WIDTH_IN,FIG_HEIGHT_IN),dpi=DPI)
    ax.imshow(-x,cmap='gray',origin='lower',extent=[0,nelx,0,nely])
    for i in range(nelx):
        for j in range(nely):
            ax.plot([i,i+1],[j+0.5,j+0.5],color='white',lw=0.5)
    ax.axis('off'); fig.savefig('results/iter_0000.png',bbox_inches='tight',pad_inches=0); plt.close(fig)
    # optimization
    for k in range(1,iters+1):
        U=FE(nelx,nely,x,theta,penal)
        c = (U.T@U)[0,0]
        compliance_hist.append(c)
        dc = np.zeros_like(x); dth = np.zeros_like(theta)
        for i in range(nelx):
            for j in range(nely):
                n1=j+i*(nely+1); n2=j+(i+1)*(nely+1)
                ed=np.array([2*n1,2*n1+1,2*n2,2*n2+1,2*n2+2,2*n2+3,2*n1+2,2*n1+3])
                Ue=U[ed,0]; Kp=KE_aniso(theta[j,i])
                dc[j,i]=-penal*x[j,i]**(penal-1)*Ue.T@Kp@Ue
                dK=(KE_aniso(theta[j,i]+1e-3)-Kp)/1e-3
                dth[j,i]=-x[j,i]**penal*Ue.T@dK@Ue
        dc=check(nelx,nely,rmin,x,dc)
        x=OC(nelx,nely,x,volfrac,dc)
        theta=filter_theta(theta+alpha_theta*dth,rmin)
        x=np.clip(x,0.001,1); theta=np.mod(theta,np.pi)
        print(f"Iter {k}: Compliance={c:.2f}, Vol={x.mean():.3f}")
        # plot frame
        fig,ax=plt.subplots(figsize=(FIG_WIDTH_IN,FIG_HEIGHT_IN),dpi=DPI)
        ax.imshow(-x,cmap='gray',origin='lower',extent=[0,nelx,0,nely])
        for i in range(nelx):
            for j in range(nely):
                if x[j,i]>0.6:
                    th=theta[j,i]; cx,cy=i+0.5,j+0.5
                    dx,dy=0.5*np.cos(th),0.5*np.sin(th)
                    ax.plot([cx-dx,cx+dx],[cy-dy,cy+dy],color='white',lw=0.5)
        ax.axis('off'); fig.savefig(f'results/iter_{k:04d}.png',bbox_inches='tight',pad_inches=0); plt.close(fig)
    # plot compliance history
    plt.figure(figsize=(6,4))
    plt.plot(range(1,iters+1), compliance_hist, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Compliance')
    plt.grid(True)
    plt.savefig('results/compliance.png', dpi=150)
    plt.close()
    # compile GIF
    files=sorted(glob.glob('results/iter_*.png'))
    frames=[Image.open(f) for f in files]
    frames[0].save('results/topopt.gif',format='GIF',save_all=True,append_images=frames[1:],duration=GIF_DURATION_MS,loop=0)

if __name__=='__main__':
    top(60,20,0.5,3.0,1.5,60)
