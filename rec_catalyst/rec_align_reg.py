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
from timing import tic, toc
matplotlib.use('Agg')


def flowplot(u, psi, flow, binning):
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
    plt.imshow(u[nz//2].real, cmap='gray')
    plt.subplot(3, 4, 10)
    plt.imshow(u[nz//2+nz//8].real, cmap='gray')

    plt.subplot(3, 4, 11)
    plt.imshow(u[:, n//2].real, cmap='gray')

    plt.subplot(3, 4, 12)
    plt.imshow(u[:, :, n//2].real, cmap='gray')
    if not os.path.exists('/data/staff/tomograms/viknik/lamino//flow/'+str(binning)+'_'+str(ntheta)+'/'):
        os.makedirs('/data/staff/tomograms/viknik/lamino//flow/' +
                    str(binning)+'_'+str(ntheta)+'/')
    plt.savefig('/data/staff/tomograms/viknik/lamino//flow/' +
                str(binning)+'_'+str(ntheta)+'/flow'+str(k))
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


if __name__ == "__main__":
    # cupy uses managed memory
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    # read data and angles
    data = dxchange.read_tiff(
        '/data/staff/tomograms/viknik/lamino/lamni-data-sorted-prealigned-cropped.tif').astype('complex64')[:, 8:-8, 8:-8]
    theta = np.load(
        '/data/staff/tomograms/viknik/lamino/angles.npy').astype('float32')/180*np.pi
    ids_bad = np.array([29,44,55,102,103,147,150,151])    
    ids_good = np.delete(np.arange(data.shape[0]),ids_bad)
    data = data[ids_good]
    theta = theta[ids_good]
    print(data.shape)

    phi = 61.18/180*np.pi
    det = data.shape[2]
    ntheta = data.shape[0]

    # normalize data for optical flow computations
    mmin = -0.5  # min data value
    mmax = 1.3  # max data value
    data[data < mmin] = mmin
    data = (data-mmin)/(mmax-mmin)

    # initial guess
    u = np.zeros([det, det, det], dtype='complex64')
    psi1 = data.copy()
    psi2 = np.zeros([3, det, det, det], dtype='complex64')
    lamd1 = np.zeros([ntheta, det, det], dtype='complex64')
    lamd2 = np.zeros([3, det, det, det], dtype='complex64')
    
    flow = np.zeros([ntheta, det, det, 2], dtype='float32')
   

    # number of ADMM iterations
    niter = 257
    # optical flow parameters
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]
    alpha = float(sys.argv[1])
    with lcg.SolverLam(det, det, det, det, ntheta, phi) as tslv:
        with dc.SolverDeform(ntheta, det, det) as dslv:
            rho1 = 0.5
            rho2 = 0.5
            
            h01 = psi1.copy()
            h02 = psi2.copy()
            
            for k in range(niter):
                # registration
                tic()
                flow = dslv.registration_flow_batch(
                    psi1, data, 0, 1, flow.copy(), pars, nproc=16)
                t1 = toc()
                tic()
                # deformation subproblem
                psi1 = dslv.cg_deform(data, psi1, flow, 4,
                                     tslv.fwd_lam(cp.array(u), cp.array(theta)).get()+lamd1/rho1, rho1, nproc=16)
                t2 = toc()
                psi2 = tslv.solve_reg(u,lamd2,rho2,alpha)   
                # tomo subproblem
                tic()
                u = tslv.cg_lam_ext(cp.array(psi1-lamd1/rho1),
                                cp.array(u), cp.array(theta), 4, rho2/rho1, cp.array(psi2-lamd2/rho2)).get()
                t3 = toc()
                h1 = tslv.fwd_lam(cp.array(u), cp.array(theta)).get()
                h2 = tslv.fwd_reg(u)
                # lambda update
                lamd1 = lamd1+rho1*(h1-psi1)
                lamd2 = lamd2+rho2*(h2-psi2)

                # checking intermediate results
                flowplot(u, psi1, flow, 0)
                if(np.mod(k, 16) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi1, flow)
                    lagr = np.zeros(7)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd1)*(h1-psi1)))
                    lagr[2] = rho1*np.linalg.norm(h1-psi1)**2
                    lagr[3] = alpha*np.sum(np.sqrt(np.real(np.sum(psi2*np.conj(psi2), 0))))
                    lagr[4] = np.sum(np.real(np.conj(lamd2*(h2-psi2))))
                    lagr[5] = rho2*np.linalg.norm(h2-psi2)**2
                    lagr[6] = np.sum(lagr[0:5])
                    print(k,lagr)
                    print('times:', t1, t2, t3)
                    sys.stdout.flush()
                    dxchange.write_tiff_stack(
                        u.real,  '/data/staff/tomograms/viknik/lamino/rec_align/tmp'+str(0)+'_'+str(ntheta)+'_'+str(alpha)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi1.real, '/data/staff/tomograms/viknik/lamino/prj_align/tmp'+str(0)+'_'+str(ntheta)+'_'+str(alpha)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho1 = update_penalty(psi1, h1, h01, rho1)
                rho2 = update_penalty(psi2, h2, h02, rho2)
                h01 = h1
                h02 = h2
                if(pars[2] > 8):
                    pars[2] -= 1
