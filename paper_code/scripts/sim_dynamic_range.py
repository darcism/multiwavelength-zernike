##############################
# Related to Figure 2
# Simulate a dataset of reconstructions for the monochromatic ZWFS, vZWFS and multiwavelength ZWFS, vZWFS 
# Generates a set of OPD screens and reconstructs them for various configurations of ZWFS
##############################

from hcipy import *
import numpy as np
from utils import make_opd_screens, sim_recon_monoc, sim_recon_mwzwfs, sim_recon_vzwfs, sim_recon_mwvzwfs
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(222)

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

# Generate a set of OPD error screens
n_samples = 500
min_rms = 0 # nm
max_rms = 250 # nm

min_power_law_exponent = -3
max_power_law_exponent = -1

input_opd_coeffs_set = make_opd_screens(aperture, pupil_diameter, Z_inv, n_samples, min_rms, max_rms, min_exponent=-3, max_exponent=-1, rms_log_space=False)

#############
# Monochromatic scalar
#############
# 600 nm
wavelength = 600e-9
phase_step = np.pi/2
phase_dot_diameter = 2

est_coeffs_monoc = np.zeros((n_samples, num_modes))

nphot=1e7

for k in tqdm(np.arange(n_samples)):

	input_opd_coeffs = input_opd_coeffs_set[k,:]

	theta = sim_recon_monoc(input_opd_coeffs, aperture, Z, phase_step, phase_dot_diameter, pupil_diameter, wavelength, nphot, xtol=1e-2, maxiter=50, add_noise=False)

	est_coeffs_monoc[k] = theta.x

np.save(f'{int(wavelength*1e9)}nm_phasestep{phase_step:.2f}_dotdia{phase_dot_diameter:.2f}_est_coeffs_monoc'.replace('.', '-') + '.npy', est_coeffs_monoc)

# 1000 nm
wavelength = 1000e-9
phase_step = np.pi/2
phase_dot_diameter = 1.2

est_coeffs_monoc = np.zeros((n_samples, num_modes))

nphot=1e7

for k in tqdm(np.arange(n_samples)):

	input_opd_coeffs = input_opd_coeffs_set[k,:]

	theta = sim_recon_monoc(input_opd_coeffs, aperture, Z, phase_step, phase_dot_diameter, pupil_diameter, wavelength, nphot, xtol=1e-2, maxiter=50, add_noise=False)

	est_coeffs_monoc[k] = theta.x

np.save(f'{int(wavelength*1e9)}nm_phasestep{phase_step:.2f}_dotdia{phase_dot_diameter:.2f}_est_coeffs_monoc'.replace('.', '-') + '.npy', est_coeffs_monoc)

#############
# Multiwavelength scalar
#############
wavelengths = np.array([600e-9, 1000e-9])
design_wavelength = 600e-9

nphot=1e7

design_phase_shift = 5*np.pi/2 # [rad]
design_diameter = 2 # [l/D]

est_coeffs_mwzwfs = np.zeros((n_samples, num_modes))

for k in tqdm(np.arange(n_samples)):

	input_opd_coeffs = input_opd_coeffs_set[k,:]

	theta = sim_recon_mwzwfs(input_opd_coeffs, aperture, Z, design_phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=150, add_noise=False)

	est_coeffs_mwzwfs[k] = theta.x

np.save(f'{int(wavelengths[0]*1e9)}nm_{int(wavelengths[1]*1e9)}nm_phasestep{design_phase_shift:.2f}_dotdia{design_diameter:.2f}_est_coeffs_mwzwfs'.replace('.', '-') + '.npy', est_coeffs_mwzwfs)

############
# Monochromatic vector
############
# 600 nm
wavelength = 600e-9
phase_step = np.pi/2
phase_dot_diameter = 2
nphot=1e7

est_coeffs_vzwfs = np.zeros((n_samples, num_modes))

for k in tqdm(np.arange(n_samples)):

	input_opd_coeffs = input_opd_coeffs_set[k,:]

	theta = sim_recon_vzwfs(input_opd_coeffs, aperture, Z, phase_step, phase_dot_diameter, pupil_diameter, wavelength, nphot, xtol=1e-2, maxiter=150, add_noise=False)

	est_coeffs_vzwfs[k] = theta.x

np.save(f'{int(wavelength*1e9)}nm_phasestep{phase_step:.2f}_dotdia{phase_dot_diameter:.2f}_est_coeffs_vzwfs'.replace('.', '-') + '.npy', est_coeffs_vzwfs)

# 1000 nm
wavelength = 1000e-9
phase_step = np.pi/2
phase_dot_diameter = 1.2
nphot=1e7

est_coeffs_vzwfs = np.zeros((n_samples, num_modes))

for k in tqdm(np.arange(n_samples)):

	input_opd_coeffs = input_opd_coeffs_set[k,:]

	theta = sim_recon_vzwfs(input_opd_coeffs, aperture, Z, phase_step, phase_dot_diameter, pupil_diameter, wavelength, nphot, xtol=1e-2, maxiter=150, add_noise=False)

	est_coeffs_vzwfs[k] = theta.x

np.save(f'{int(wavelength*1e9)}nm_phasestep{phase_step:.2f}_dotdia{phase_dot_diameter:.2f}_est_coeffs_vzwfs'.replace('.', '-') + '.npy', est_coeffs_vzwfs)

############
# Multiwavelength vector
############
wavelengths = np.array([600e-9, 1000e-9])
design_wavelength = 600e-9

phase_shift = np.pi/2 # [rad]
design_diameter = 2 # [l/D]

est_coeffs_mwvzwfs = np.zeros((n_samples, num_modes))

for k in tqdm(np.arange(n_samples)):

	input_opd_coeffs = input_opd_coeffs_set[k,:]

	theta = sim_recon_mwvzwfs(input_opd_coeffs, aperture, Z, phase_step, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=150, add_noise=False)

	est_coeffs_mwvzwfs[k] = theta.x

np.save(f'{int(wavelengths[0]*1e9)}nm_{int(wavelengths[1]*1e9)}nm_phasestep{phase_shift:.2f}_dotdia{design_diameter:.2f}_est_coeffs_mwvzwfs'.replace('.', '-') + '.npy', est_coeffs_mwvzwfs)