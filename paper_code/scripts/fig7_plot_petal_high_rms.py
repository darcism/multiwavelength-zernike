##############################
# Reproduce Figure 7 on reconstruction large petal error
# Note: PC that ran paper simulations consistenly reproduces figure 
# 		However local pc shows variations between runs although still produces a good fit
# 		Be aware that this might be possible
##############################

import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from utils import make_elt_like_aperture, sim_recon_vzwfs_pu

# Optical system
pupil_diameter = 1 # [m]
focal_length = 1 # [m]

# Detector
num_pix = 128

# Pupil
pupil_grid = make_pupil_grid(dims=num_pix, diameter=pupil_diameter*1.1)

# Optical system
wavelengths = np.array([600e-9, 700e-9])
design_wavelength = wavelengths[0]
phase_shift = np.pi/2 # [rad]
design_diameter = 2 # [l/D]
nphot = 1e7

# Aperture
num_segment_rings = 3
gap_size = 8e-3
spider_width = 0.02
n_spiders = 6
offset_spiders = 0
eltapg, segments = make_elt_like_aperture(pupil_diameter=pupil_diameter, num_segment_rings=num_segment_rings, gap_size=gap_size, spider_width=spider_width, n_spiders=n_spiders, offset_spiders=offset_spiders, with_spiders=True, return_segments=True)
aperture = evaluate_supersampled(eltapg, pupil_grid, 1)

# Input error
petal_coeffs = np.zeros(n_spiders)
petal_coeffs[0] = 0
petal_coeffs[1] = 620
petal_coeffs[2] = -1700
petal_coeffs[3] = 1870
petal_coeffs[4] = -1250
petal_coeffs[5] = 2000

# Petal basis
spider_angles = (360/n_spiders) * np.arange(n_spiders) + offset_spiders
spider_angles = np.append(spider_angles, 360)

angle = Field(np.arctan2(aperture.grid.y, aperture.grid.x), pupil_grid)
angle[angle<0] += 2*np.pi
angle = np.rad2deg(angle)
petal_tf = aperture.grid.zeros((n_spiders,))
for i in np.arange(n_spiders):
	
	petal = (angle > spider_angles[i]) * (angle <= spider_angles[(i+1)%(n_spiders+1)]) * aperture

	petal_tf[i] = petal

petal_basis = ModeBasis(petal_tf.T)

# Reconstruct
Z = petal_basis.transformation_matrix
petal_opd = Field(np.matmul(petal_basis.transformation_matrix, petal_coeffs*1e-9), pupil_grid)

ref_seg_id=0
c, wo0, wo1 = sim_recon_vzwfs_pu(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=50, add_noise=False)
c += (petal_coeffs[0] - c[0])

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(15,3.4))
plt.subplot(131)
imshow_field(petal_opd * 1e9, vmin=-2100, vmax=2100, cmap='RdBu'), plt.title('Input OPD [nm]'), plt.colorbar(), plt.xlabel('(a)')
plt.xticks([], []), plt.yticks([], [])
plt.subplot(132)
imshow_field(Field(np.matmul(petal_basis.transformation_matrix, c), pupil_grid), vmin=-2100, vmax=2100, cmap='RdBu'), plt.title('Estimated OPD [nm]'), plt.colorbar(), plt.xlabel('(b)')
plt.xticks([], []), plt.yticks([], [])
plt.subplot(133)
imshow_field(Field(np.matmul(petal_basis.transformation_matrix, petal_coeffs - c), pupil_grid), vmin=-0.25, vmax=0.25, cmap='RdBu'), plt.title('Residual [nm]'), plt.colorbar(), plt.xlabel('(c)')
plt.xticks([], []), plt.yticks([], [])

plt.show()