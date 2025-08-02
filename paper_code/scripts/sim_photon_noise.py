##############################
# Related to Figure 4
# Simulate a dataset of an OPD screen reconstruction under various photon levels
# Reconstruction for a monochromatic and two multiwavelength configurations of the ZWFS
##############################

import numpy as np
from tqdm import tqdm
from hcipy import *
from utils import make_opd_error, sim_recon_monoc, sim_recon_mwzwfs

# Simulation settings
nphot_array = np.array([5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
n_samples = 100
max_iter = 50
xtol = 1e-3

# Optical system
pupil_diameter = 1 # [m]
focal_length = 1 # [m]

# Detector
num_pix = 64

# Pupil
pupil_grid = make_pupil_grid(dims=num_pix, diameter=pupil_diameter*1.1)
aperture = make_circular_aperture(diameter=pupil_diameter)(pupil_grid)

# Zernike basis
num_modes = 75
zernike_basis = make_zernike_basis(num_modes=num_modes, D=pupil_diameter, grid=pupil_grid, starting_mode=4)
Z = zernike_basis.transformation_matrix
Z_inv = inverse_tikhonov(Z, 1e-10)

# Input error
np.random.seed(4)
exponent = -2
input_rms = 20
ptt_b = make_zernike_basis(num_modes=3, D=pupil_diameter, grid=pupil_grid, starting_mode=1) # remove piston, tip, tilt
error = make_opd_error(exponent=exponent, rms=input_rms, aperture=aperture, pupil_diameter=pupil_diameter, remove_modes=ptt_b)

input_opd_coeffs = np.matmul(Z_inv, error)
rms = np.sqrt(np.sum(np.square(input_opd_coeffs)))
input_opd_coeffs *= (input_rms / rms)

##################
# Monochromatic scalar
##################
l = 600e-9
phase_shift = np.pi/2
dot_diam = 2

rms_errors_zwfs = np.zeros((len(nphot_array), n_samples))

for k, nphot in enumerate(nphot_array):
	for i in tqdm(np.arange(n_samples)):

		theta = sim_recon_monoc(input_opd_coeffs, aperture, Z, phase_step=phase_shift, phase_dot_diameter=dot_diam, pupil_diameter=pupil_diameter, wavelength=l, nphot=int(nphot), xtol=xtol, maxiter=max_iter, add_noise=True)

		rms = np.sqrt(np.sum(np.square(theta.x - input_opd_coeffs)))
		rms_errors_zwfs[k, i] = rms

np.save('rms_errors_zwfs.npy', rms_errors_zwfs)

##################
# Multiwavelength scalar
##################
# 600, 1000 nm
l0 = 600e-9
dl = 400e-9
wavelengths = np.array([l0, l0+dl])
phase_shift_l0 = 5*np.pi/2
dot_diam_l0 = 2

rms_errors_mwzwfs = np.zeros((len(nphot_array), n_samples))

for k, nphot in enumerate(nphot_array):
	for i in tqdm(np.arange(n_samples)):

		theta = sim_recon_mwzwfs(input_opd_coeffs, aperture, Z, design_phase_shift=phase_shift_l0, design_diameter=dot_diam_l0, design_wavelength=l0, wavelengths=wavelengths, nphot=int(nphot), pupil_diameter=pupil_diameter, xtol=xtol, maxiter=max_iter, add_noise=True)

		rms = np.sqrt(np.sum(np.square(theta.x - input_opd_coeffs)))
		rms_errors_mwzwfs[k, i] = rms

np.save('rms_errors_mwzwfs_600_1000.npy', rms_errors_mwzwfs)

# 428, 600 nm
l0 = 600e-9
dl = -172e-9
wavelengths = np.array([l0, l0+dl])
phase_shift_l0 = 5*np.pi/2
dot_diam_l0 = 2

rms_errors_mwzwfs = np.zeros((len(nphot_array), n_samples))

for k, nphot in enumerate(nphot_array):
	for i in tqdm(np.arange(n_samples)):

		theta = sim_recon_mwzwfs(input_opd_coeffs, aperture, Z, design_phase_shift=phase_shift_l0, design_diameter=dot_diam_l0, design_wavelength=l0, wavelengths=wavelengths, nphot=int(nphot), pupil_diameter=pupil_diameter, xtol=xtol, maxiter=max_iter, add_noise=True)

		rms = np.sqrt(np.sum(np.square(theta.x - input_opd_coeffs)))
		rms_errors_mwzwfs[k, i] = rms

np.save('rms_errors_mwzwfs_428_600.npy', rms_errors_mwzwfs)