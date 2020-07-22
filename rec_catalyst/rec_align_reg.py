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
# cupy uses managed memory
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)


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
    if not os.path.exists('flow_new_'+str(ntheta)+'/'):
        os.makedirs('flow_new_'+str(ntheta)+'/')
    plt.savefig('flow_new_'+str(ntheta)+'/flow'+str(k))
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

    # read data and angles
    data = dxchange.read_tiff(
        '/home/beams0/VNIKITIN/lamino_doga/lamalign/data/rec_new.tiff').astype('float32')#[:, ::4, ::4]
    theta = np.load(
        '/home/beams0/VNIKITIN/lamino_doga/lamalign/data/angle.npy').astype('float32')/180*np.pi
    #ids_bad = np.array([29,44,56,102,152])
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
    psi1 = data.copy()
    psi2 = np.zeros([3, det, det, det], dtype='float32')
    lamd1 = np.zeros([ntheta, det, det], dtype='float32')
    lamd2 = np.zeros([3, det, det, det], dtype='float32')

    flow = np.zeros([ntheta, det, det, 2], dtype='float32')

    # optical flow parameters
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]
    niter = 257
    alpha = 1.3e-8

    with lcg.SolverLam(det, det, det, det, ntheta, phi, 1e-3) as tslv:
        with dc.SolverDeform(ntheta, det, det) as dslv:
            rho1 = 0.5
            rho2 = 0.5

            h01 = psi1
            h02 = psi2

            for k in range(niter):
                # registration
                tic()
 
                flow = dslv.registration_flow_batch(
                    psi1, data, mmin, mmax, flow.copy(), pars, nproc=42)
                t1 = toc()
                tic()
                # deformation subproblem
                psi1 = dslv.cg_deform(data, psi1, flow, 4,
                                      tslv.fwd_lam(u, theta)+lamd1/rho1, rho1, nproc=42)
                t2 = toc()

                psi2 = tslv.solve_reg(u, lamd2, rho2, alpha)

                # tomo subproblem
                tic()
                u = tslv.cg_lam_ext(psi1-lamd1/rho1,
                                    u, theta, 4, rho2/rho1, psi2-lamd2/rho2, False)
                t3 = toc()

                h1 = tslv.fwd_lam(u, theta)
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
                    lagr[3] = alpha * \
                        np.sum(np.sqrt(np.real(np.sum(psi2*np.conj(psi2), 0))))
                    lagr[4] = np.sum(np.real(np.conj(lamd2*(h2-psi2))))
                    lagr[5] = rho2*np.linalg.norm(h2-psi2)**2
                    lagr[6] = np.sum(lagr[0:5])
                    print(k, rho1, rho2, lagr)
                    print('times:', t1, t2, t3)
                    sys.stdout.flush()
                    dxchange.write_tiff_stack(
                        u,  'rec_reg_align_new/tmp'+'_'+str(ntheta)+'_'+str(alpha)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi1, 'prj_reg_align_new/tmp'+'_'+str(ntheta)+'_'+str(alpha)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho1 = update_penalty(psi1, h1, h01, rho1)
                rho2 = update_penalty(psi2, h2, h02, rho2)
                h01 = h1
                h02 = h2
                if(pars[2] > 8):
                    pars[2] -= 1
