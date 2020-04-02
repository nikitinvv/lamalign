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
    lamd = np.zeros([ntheta, det, det], dtype='complex64')
    flow = np.zeros([ntheta, det, det, 2], dtype='float32')
    psi = data.copy()

    # number of ADMM iterations
    niter = 257
    # optical flow parameters
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]

    with lcg.SolverLam(det, det, det, det, ntheta, phi) as tslv:
        with dc.SolverDeform(ntheta, det, det) as dslv:
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                tic()
                flow = dslv.registration_flow_batch(
                    psi, data, 0, 1, flow.copy(), pars, nproc=21)
                t1 = toc()
                tic()
                # deformation subproblem
                psi = dslv.cg_deform(data, psi, flow, 4,
                                     tslv.fwd_lam(cp.array(u), cp.array(theta)).get()+lamd/rho, rho, nproc=21)
                t2 = toc()
                # tomo subproblem
                tic()
                u = tslv.cg_lam(cp.array(psi-lamd/rho),
                                cp.array(u), cp.array(theta), 4).get()
                t3 = toc()
                h = tslv.fwd_lam(cp.array(u), cp.array(theta)).get()
                # lambda update
                lamd = lamd+rho*(h-psi)

                # checking intermediate results
                flowplot(u, psi, flow, 0)
                if(np.mod(k, 4) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, pars[2], np.linalg.norm(flow), rho, lagr)
                    print('times:', t1, t2, t3)
                    dxchange.write_tiff_stack(
                        u.real,  '/data/staff/tomograms/viknik/lamino/rec_align/tmp'+str(0)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi.real, '/data/staff/tomograms/viknik/lamino/prj_align/tmp'+str(0)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                if(pars[2] > 8):
                    pars[2] -= 1
