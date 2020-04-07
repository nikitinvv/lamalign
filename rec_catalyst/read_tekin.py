import dxchange
import h5py
import numpy as np
f = h5py.File('/data/staff/tomograms/viknik/lamino/extracted_scan251.h5', 'r')
a1 = f['cg_obj_recon'][:]
dxchange.write_tiff(np.angle(a1),'/data/staff/tomograms/viknik/lamino/cg_obj_recon')
a2 = f['matlab_obj_recon'][:]
dxchange.write_tiff(np.angle(a2),'/data/staff/tomograms/viknik/lamino/matlab_obj_recon')
a3 = f['cg_probe_recon'][:]
dxchange.write_tiff(np.abs(a3),'/data/staff/tomograms/viknik/lamino/cg_probe_recon')
a3 = f['recprobe'][:]
dxchange.write_tiff(np.abs(a3),'/data/staff/tomograms/viknik/lamino/recprobe')