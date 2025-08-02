##############################
# Related to Figure 6
# Simulate a dataset of single petal error reconstructions
# Reconstruction using 2-wavelength phase unwrapping vs monochromatic reconstruction
##############################

import numpy as np
from hcipy import *
from tqdm import tqdm
from utils import sim_recon_mwvzwfs_dopd, make_elt_like_aperture, sim_recon_vzwfs_pu

# Petal amplitudes
num_amps = 40
opd_amplitudes = np.linspace(-2500, 2500, num=num_amps)
ref_seg_id = 0 # Reference petal number
petal_id = 1 # Petal number to deform

# Optical system
pupil_diameter = 1 # [m]
focal_length = 1 # [m]

# Detector
num_pix = 128

# Pupil
pupil_grid = make_pupil_grid(dims=num_pix, diameter=pupil_diameter*1.1)

# Wavelengths
wavelengths = np.array([600e-9, 700e-9])
nphot = 1e7

# Mask
design_wavelength = wavelengths[0]
phase_shift = np.pi/2 # [rad]
design_diameter = 2 # [l/D]

# Aperture
num_segment_rings = 3
gap_size = 8e-3
spider_width = 0.02
n_spiders = 6
offset_spiders = 0
eltapg, segments = make_elt_like_aperture(pupil_diameter=pupil_diameter, num_segment_rings=num_segment_rings, gap_size=gap_size, spider_width=spider_width, n_spiders=n_spiders, offset_spiders=offset_spiders, with_spiders=True, return_segments=True)
aperture = evaluate_supersampled(eltapg, pupil_grid, 1)

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
Z = petal_basis.transformation_matrix

# Two-wavelength phase unwrapping simulation
recon_petal_coeffs_vzwfs_gd = np.zeros((num_amps, n_spiders))
for i, opd_amp in tqdm(enumerate(opd_amplitudes)):

	petal_coeffs = np.zeros(n_spiders)
	petal_coeffs[petal_id] = opd_amp

	# Gradient descent
	c, wo0, wo1 = sim_recon_vzwfs_pu(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=50, add_noise=False)
	c += (petal_coeffs[0] - c[0])

	recon_petal_coeffs_vzwfs_gd[i] = c[petal_id]
	
np.save(f'{int(wavelengths[0]*1e9)}nm_{int(wavelengths[1]*1e9)}nm_phasestep{phase_shift:.2f}_dotdia{design_diameter:.2f}_petal_recons_coeffs_vzwfs_gd'.replace('.', '-') + '.npy', recon_petal_coeffs_vzwfs_gd)  

# Monochromatic simulation
recon_petal_coeffs_vzwfs_m = np.zeros((num_amps, n_spiders))
for i, opd_amp in tqdm(enumerate(opd_amplitudes)):

	petal_coeffs = np.zeros(n_spiders)
	petal_coeffs[petal_id] = opd_amp

	cc = sim_recon_mwvzwfs_dopd(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, np.array([wavelengths[1]]), nphot, pupil_diameter, add_noise=False)

	recon_petal_coeffs_vzwfs_m[i] = (cc.x[petal_id] - cc.x[0])

np.save(f'{int(wavelengths[1]*1e9)}nm_phasestep{phase_shift:.2f}_dotdia{design_diameter:.2f}_recon_petal_coeffs_vzwfs_m'.replace('.', '-') + '.npy', recon_petal_coeffs_vzwfs_m)