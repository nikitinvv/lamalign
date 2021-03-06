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
matplotlib.use('Agg')


def flowplot(u, psi, flow, binning):
    [ntheta, nz, n] = psi.shape

    plt.figure(figsize=(20, 14))
    plt.subplot(3, 4, 1)
    plt.imshow(psi[ntheta//4], cmap='gray')

    plt.subplot(3, 4, 2)
    plt.imshow(psi[ntheta//2], cmap='gray')
    plt.subplot(3, 4, 3)
    plt.imshow(psi[3*ntheta//4], cmap='gray')

    plt.subplot(3, 4, 4)
    plt.imshow(psi[-1], cmap='gray')

    plt.subplot(3, 4, 5)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//4]), cmap='gray')

    plt.subplot(3, 4, 6)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')

    plt.subplot(3, 4, 7)
    plt.imshow(dc.flowvis.flow_to_color(flow[3*ntheta//4]), cmap='gray')
    plt.subplot(3, 4, 8)
    plt.imshow(dc.flowvis.flow_to_color(flow[-1]), cmap='gray')

    plt.subplot(3, 4, 9)
    plt.imshow(u[nz//2], cmap='gray')
    plt.subplot(3, 4, 10)
    plt.imshow(u[nz//2+nz//8], cmap='gray')

    plt.subplot(3, 4, 11)
    plt.imshow(u[:, n//2], cmap='gray')

    plt.subplot(3, 4, 12)
    plt.imshow(u[:, :, n//2], cmap='gray')
    if not os.path.exists('flow_new'+'_'+str(ntheta)+'/'):
        os.makedirs('flow_new' + '_'+str(ntheta)+'/')
    plt.savefig('flow_new' + '_'+str(ntheta)+'/flow'+str(k))
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
        '/home/beams0/VNIKITIN/lamino_doga/lamalign/data/rec_new.tiff').astype('float32')#[:, ::4, ::4]
    theta = np.load(
        '/home/beams0/VNIKITIN/lamino_doga/lamalign/data/angle.npy').astype('float32')/180*np.pi
    phi = 61.18/180*np.pi
    det = data.shape[2]
    ntheta = data.shape[0]

    # normalize data for optical flow computations
    mmin = -0.3  # min data value
    mmax = 2  # max data value
    data[data < mmin] = mmin
    data[data > mmax] = mmax
    #data = (data-mmin)/(mmax-mmin)

    # initial guess
    u = np.zeros([det, det, det], dtype='float32')
    lamd = np.zeros([ntheta, det, det], dtype='float32')
    flow = np.zeros([ntheta, det, det, 2], dtype='float32')
    psi = data.copy()

    # optical flow parameters
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]
    niter = 257
    with lcg.SolverLam(det, det, det, det, ntheta, phi,float(sys.argv[1])) as tslv:
        with dc.SolverDeform(ntheta, det, det) as dslv:
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                tic()
                flow = dslv.registration_flow_batch(
                    psi, data, mmin, mmax, flow.copy(), pars, nproc=42)
                t1 = toc()
                tic()
                # deformation subproblem
                psi = dslv.cg_deform(data, psi, flow, 4,
                                     tslv.fwd_lam(u, theta)+lamd/rho, rho, nproc=42)
                t2 = toc()
                # tomo subproblem
                tic()
                u = tslv.cg_lam(psi-lamd/rho,
                                u, theta, 4, False)
                t3 = toc()
                h = tslv.fwd_lam(u, theta)
                # lambda update
                lamd = lamd+rho*(h-psi)

                # checking intermediate results
                flowplot(u, psi, flow, 0)
                if(np.mod(k, 8) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k,  rho, pars[2], np.linalg.norm(flow), rho, lagr)
                    print('times:', t1, t2, t3)
                    dxchange.write_tiff_stack(
                        u,  'rec_align_new/tmp'+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi, 'prj_align_new/tmp'+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                if(pars[2] > 8):
                    pars[2] -= 1
