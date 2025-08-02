##############################
# Reproduce Figure 8 on phase unwrapping noise
# Note: Due to randomness in seed results can vary but do not change conclusions
##############################

import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from tqdm import tqdm
from utils import sim_recon_mwvzwfs_dopd, wrapped_opd, sim_recon_vzwfs_pu, make_elt_like_aperture

petal_id = 1 # Petal number to deform
ref_seg_id = 0 # Reference petal number
n_spiders = 6
petal_coeffs = np.zeros(n_spiders)
petal_coeffs[petal_id] = 150 # nm

# np.random.seed(3) # To fix result

# Optical system
wavelengths = np.array([600e-9, 700e-9])
design_wavelength = wavelengths[0]
phase_shift = np.pi/2 # [rad]
design_diameter = 2 # [l/D]

pupil_diameter = 1 # [m]
focal_length = 1 # [m]
num_pix = 128

pupil_grid = make_pupil_grid(dims=num_pix, diameter=pupil_diameter*1.1)

# Aperture
num_segment_rings = 3
gap_size = 8e-3
spider_width = 0.02
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

# Reconstruct
nphot_array = np.array([5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
n_samples = 100

recon_pu = np.zeros((len(nphot_array), n_samples))
recon_w0 = np.zeros((len(nphot_array), n_samples))
recon_w1 = np.zeros((len(nphot_array), n_samples))

Z = petal_basis.transformation_matrix
for i, nph in tqdm(enumerate(nphot_array)):
	for j in np.arange(n_samples):
		# Gradient descent on separate wavelengths
		th0 = sim_recon_mwvzwfs_dopd(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, np.array([wavelengths[0]]), int(nph), pupil_diameter, add_noise=True)
		th1 = sim_recon_mwvzwfs_dopd(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, np.array([wavelengths[1]]), int(nph), pupil_diameter, add_noise=True)

		recon_w0[i, j] = wrapped_opd(th0.x[petal_id] - th0.x[0], wavelengths[0]*1e9)
		recon_w1[i, j] = wrapped_opd(th1.x[petal_id] - th1.x[0], wavelengths[1]*1e9)

		# Gradient descent with two-wavelength phase unwrapping
		c, wo0, wo1 = sim_recon_vzwfs_pu(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, wavelengths, int(nph), pupil_diameter, xtol=1e-2, maxiter=50, add_noise=True)
		c += (petal_coeffs[0] - c[0])

		recon_pu[i, j] = c[petal_id]

# Plot
plt.rcParams.update({'font.size': 16})
blue = '#4477AA'
red = '#EE6677'
plt.figure(figsize=(10,8))

yest_w1 = abs(np.average(recon_w1, axis=1) - petal_coeffs[petal_id])
yerr_w1 = np.sqrt(np.var(recon_w1, axis=1))
plt.plot(nphot_array, yest_w1, c=red, marker='.', markersize=10, label='Mono-GD (700 nm)')
plt.fill_between(nphot_array, yest_w1 - yerr_w1, yest_w1 + yerr_w1, color=red, alpha=0.2)

yest_pu = abs(np.average(recon_pu, axis=1) - petal_coeffs[petal_id])
yerr_pu = np.sqrt(np.var(recon_pu, axis=1))
plt.plot(nphot_array, yest_pu, c=blue, marker='.', markersize=10, label='Phase unwrapped (600, 700 nm)')
plt.fill_between(nphot_array, yest_pu - yerr_pu, yest_pu + yerr_pu, color=blue, alpha=0.2)

plt.xlabel('$N_{phot}$ per wavelength'), plt.ylabel('Residual [nm]'), plt.xlim([5000, 80000]), plt.legend()
plt.title('Input 150 nm petal error')

plt.show()