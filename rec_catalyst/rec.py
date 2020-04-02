import numpy as np
import lamcg as lcg
import cupy as cp
import dxchange

if __name__ == "__main__":
    # cupy uses managed memory 
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    data = dxchange.read_tiff('/data/staff/tomograms/viknik/lamino/lamni-data-sorted-prealigned-cropped.tif').astype('complex64')[:,8:-8,8:-8]
    theta = np.load('/data/staff/tomograms/viknik/lamino/angles.npy').astype('float32')/180*np.pi
    phi = 61.18/180*np.pi
    mmin = -0.5 
    mmax = 1.3
    data[data<mmin]=mmin
    data=(data-mmin)/(mmax-mmin)
    det = data.shape[2]
    ntheta = data.shape[0]

    # reconstruct with the cg solver
    with lcg.SolverLam(det, det, det, det, ntheta, phi) as slv:
        theta_gpu = cp.array(theta)    
        data_gpu = cp.array(data)
        init_gpu = cp.zeros([det,det,det],dtype='complex64')
        rec = slv.cg_lam(data_gpu,init_gpu,theta_gpu,128)    
        dxchange.write_tiff_stack(rec.get().real,'/data/staff/tomograms/viknik/lamino/cg/r',overwrite=True)
        

    