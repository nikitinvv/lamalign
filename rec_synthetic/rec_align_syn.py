import numpy as np
import lamcg as lcg
import cupy as cp
import deformcg as dc
import dxchange
from timing import *
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic,toc
matplotlib.use('Agg')



def myplot(u, psi, flow, binning):
    [ntheta, nz, n] = psi.shape

    plt.figure(figsize=(20, 14))
    plt.subplot(3, 4, 1)
    plt.imshow(psi[ntheta//4].real, cmap='gray')

    plt.subplot(3, 4, 2)
    plt.imshow(psi[ntheta//2].real, cmap='gray')
    plt.subplot(3, 4, 3)
    plt.imshow(psi[3*ntheta//4].real, cmap='gray')

    plt.subplot(3, 4, 4)
    plt.imshow(psi[-1].real, cmap='gray')

    plt.subplot(3, 4, 5)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//4]), cmap='gray')

    plt.subplot(3, 4, 6)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')

    plt.subplot(3, 4, 7)
    plt.imshow(dc.flowvis.flow_to_color(flow[3*ntheta//4]), cmap='gray')
    plt.subplot(3, 4, 8)
    plt.imshow(dc.flowvis.flow_to_color(flow[-1]), cmap='gray')

    plt.subplot(3, 4, 9)
    plt.imshow(u[nz//2].real,cmap='gray')
    plt.subplot(3, 4, 10)
    plt.imshow(u[nz//2+nz//8].real,cmap='gray')

    plt.subplot(3, 4, 11)
    plt.imshow(u[:, n//2].real,cmap='gray')

    plt.subplot(3, 4, 12)
    plt.imshow(u[:, :, n//2].real,cmap='gray')
    if not os.path.exists('/data/staff/tomograms/viknik/lamino//flow/'+str(binning)+'_'+str(ntheta)+'/'):
        os.makedirs('/data/staff/tomograms/viknik/lamino//flow/'+str(binning)+'_'+str(ntheta)+'/')
    plt.savefig('/data/staff/tomograms/viknik/lamino//flow/'+str(binning)+'_'+str(ntheta)+'/flow'+str(k))
    plt.close()



def update_penalty(psi, h, h0, rho):
    # rho
    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho*(h-h0))**2
    if (r > 10*s):
        rho *= 2
    elif (s > 10*r):
        rho *= 0.5
    return rho

cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

binning = 1
data = dxchange.read_tiff('/data/staff/tomograms/viknik/lamino/lamni-data-sorted-prealigned-cropped.tif').astype('complex64')[:,::pow(2,binning),::pow(2,binning)]
theta = np.load('/data/staff/tomograms/viknik/lamino/angles.npy')/180*np.pi
phi = 61.18/180*np.pi
[x,y] = np.meshgrid(np.arange(-data.shape[2]//2,data.shape[2]//2),np.arange(-data.shape[2]//2,data.shape[2]//2))
x=x*2/data.shape[2]
y=y*2/data.shape[2]
circ = x**2+y**2<1
data= (data+0.3)*circ
det = data.shape[2]
ntheta = data.shape[0]
#print(theta)
theta_gpu = cp.array(np.array(theta),order='C')    
#data_gpu = cp.array(data)#slv.fwd_lam(f_gpu,theta_gpu)

u = np.zeros([det, det, det], dtype='complex64')
psi = data.copy()
lamd = np.zeros([ntheta, det, det], dtype='complex64')
flow = np.zeros([ntheta, det, det, 2], dtype='float32')
niter=128
# optical flow parameters
pars = [0.5, 0, 256, 4, 5, 1.1, 0]
mmin = 0.00 # background
mmax = 1.5
data[data<mmin]=mmin
data=(data-mmin)/(mmax-mmin)
shift = (np.random.random([ntheta,2])-.5)*5
#exit()
f=-dxchange.read_tiff('delta-chip-256.tiff').astype('complex64')#[::2,::2,::2]    
print(theta)
with lcg.SolverLam(det, det, det, det, ntheta, phi) as tslv:
    #init_gpu = cp.zeros([det,det,det],dtype='complex64')
    data = tslv.fwd_lam(cp.array(f),theta_gpu).get()
    dxchange.write_tiff_stack(data.real,'/data/staff/tomograms/viknik/lamino/datachip/r',overwrite=True)
    exit()
    #rec = slv.cg_lam(data_gpu,init_gpu,theta_gpu,32)    
    with dc.SolverDeform(ntheta, det, det) as dslv:
            data = dslv.apply_shift_batch(data, shift)
            psi=data.copy()
            mmin = np.min(data) # background
            mmax = np.max(data)
            data=(data-mmin)/(mmax-mmin)
            
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                tic()
                flow = dslv.registration_flow_batch(psi, data,0,1,flow.copy(), pars,nproc=42)
                t1=toc()
                tic()
                # deformation subproblem
                psi = dslv.cg_deform(data, psi, flow, 4,
                                     tslv.fwd_lam(cp.array(u),theta_gpu).get()+lamd/rho, rho,nproc=42)
                t2=toc()
                # tomo subproblem                
                tic()
                u = tslv.cg_lam(cp.array(psi-lamd/rho), cp.array(u),theta_gpu, 4).get()
                t3=toc()
                h = tslv.fwd_lam(cp.array(u),theta_gpu).get()
                # lambda update
                lamd = lamd+rho*(h-psi)

                # checking intermediate results
                myplot(u, psi, flow, binning)
                if(np.mod(k, 1) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, pars[2], np.linalg.norm(flow), rho, lagr, t1, t2, t3)                    
                    dxchange.write_tiff_stack(
                        u.real,  '/data/staff/tomograms/viknik/lamino/rec_align/tmp'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r',overwrite=True)
                    # dxchange.write_tiff_stack(
                        # psi1.real, 'tmp2'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                if(pars[2]>8):
                    pars[2] -= 1

    

   