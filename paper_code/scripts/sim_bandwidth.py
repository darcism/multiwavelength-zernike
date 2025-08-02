##############################
# Related to Figure 5
# Simulate a dataset of reconstructions for the monochromatic vZWFS and multiwavelength vZWFS 
# for various bandwidth sizes under photon noise
##############################

import numpy as np
from hcipy import *
from scipy.optimize import minimize
from tqdm import tqdm
from utils import vzwfs_polych_model, cost_function_v, calc_grad_v, sim_recon_mwvzwfs, make_opd_error

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

# OPD error
np.random.seed(4)
exponent = -2
input_rms = 20
ptt_b = make_zernike_basis(num_modes=3, D=pupil_diameter, grid=pupil_grid, starting_mode=1) # remove piston, tip, tilt
error = make_opd_error(exponent=exponent, rms=input_rms, aperture=aperture, pupil_diameter=pupil_diameter, remove_modes=ptt_b)

input_opd_coeffs = np.matmul(Z_inv, error)
rms = np.sqrt(np.sum(np.square(input_opd_coeffs)))
input_opd_coeffs *= (input_rms / rms)

# Mask parameters
design_wavelength = 600e-9
phase_shift = np.pi/2 # [rad]
design_diameter = 2 # [l/D]

# Bandwidth
bw = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) / 100 # %

center_l = 600e-9
n_wavelengths = 10

# Set total number of photons
nphot_nm = 100
add_noise = True

n_samples = 50

# Simulate
rms_error_vzwfs = np.zeros((len(bw), n_samples))
rms_error_mwvzwfs = np.zeros((len(bw), n_samples))

for k, bw_perc in tqdm(enumerate(bw)):

	if bw_perc == 0:
		wavelengths = np.array([center_l])
		nphot_i = np.array([nphot_tot])
	else:
		# Wavelength sampling
		lmin = center_l - (center_l * (bw_perc/2))
		lmax = center_l + (center_l * (bw_perc/2))

		wavelengths, dl = np.linspace(lmin, lmax, n_wavelengths, retstep=True)

		# flat spectrum
		nphot_tot = nphot_nm * center_l * 1e9 * (bw_perc)
		nphot_i = (nphot_tot/n_wavelengths)*np.ones(n_wavelengths)

	for n in np.arange(n_samples):
		# vzwfs
		zwfs_models, Is_meas, equiv_wavelength, nphot_equiv = vzwfs_polych_model(input_opd_coeffs, aperture, Z, phase_shift, design_diameter, design_wavelength, pupil_diameter, wavelengths, nphot_i, add_noise=add_noise)
		x0 = np.zeros(np.shape(Z)[1])
		theta = minimize(cost_function_v, x0, args=(Is_meas, Z, aperture, zwfs_models, equiv_wavelength, nphot_equiv), method='Newton-CG', jac=calc_grad_v, options={'maxiter':150, 'disp':False, 'xtol':1e-2})
		rms_error_vzwfs[k, n] = np.sqrt(np.sum(np.square(input_opd_coeffs - theta.x)))

		# mwvzwfs
		theta = sim_recon_mwvzwfs(input_opd_coeffs, aperture, Z, phase_shift, design_diameter, design_wavelength, wavelengths, nphot_i[0], pupil_diameter, xtol=1e-2, maxiter=150, add_noise=add_noise)
		rms_error_mwvzwfs[k, n] = np.sqrt(np.sum(np.square(input_opd_coeffs - theta.x)))

np.save(f'rms_error_vzwfs_l0{int(center_l*1e9)}_nlambda{n_wavelengths}_nphotnm{nphot_nm}_{n_samples}samples_{len(bw)}bwsamples'.replace('.', '-') + '.npy', rms_error_vzwfs)
np.save(f'rms_error_mwvzwfs_l0{int(center_l*1e9)}_nlambda{n_wavelengths}_nphotnm{nphot_nm}_{n_samples}sample_{len(bw)}bwsampless'.replace('.', '-') + '.npy', rms_error_mwvzwfs)