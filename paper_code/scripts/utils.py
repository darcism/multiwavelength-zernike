##############################
# Functions used for simulation
##############################

from hcipy import *
import numpy as np
from scipy.optimize import minimize

# Input OPD coefficients to output wavefront of ZWFS
def zwfs_forward_opd(zcoeffs, Z, aperture, zwfs, wavelength, nphot):
	
	# Electric field corresponding to mode coefficients
	E_in = aperture * np.exp(1j*(2*np.pi/wavelength)*np.matmul(Z,zcoeffs*1e-9))

	# HCIpy wavefront
	wf_in = Wavefront(Field(E_in, aperture.grid), wavelength)
	wf_in.total_power = nphot
	
	# Wavefront at output ZWFS
	wf_out = zwfs.forward(wf_in)

	return wf_out

#########################
# Dynamic range
#########################

# Make OPD error with certain power law and RMS
def make_opd_error(exponent, rms, aperture, pupil_diameter, remove_modes=None):
	opd_error_init = make_power_law_error(aperture.grid, ptv=1, diameter=pupil_diameter, exponent=exponent, aperture=aperture, remove_modes=remove_modes)
	rms_init = np.sqrt(np.average(np.square(opd_error_init[aperture>0])))
	opd_error = (rms/rms_init) * opd_error_init
	return opd_error

# Make set of OPD errors with certain range of power law exponents and rms values
def make_opd_screens(aperture, pupil_diameter, Z_inv, n_samples, min_rms, max_rms, min_exponent, max_exponent, rms_log_space=True):

	min_power_law_exponent = min_exponent
	max_power_law_exponent = max_exponent

	input_opd_coeffs_set = np.zeros((n_samples, np.shape(Z_inv)[0]))

	ptt_b = make_zernike_basis(num_modes=3, D=pupil_diameter, grid=aperture.grid, starting_mode=1) # remove piston, tip, tilt

	for k in np.arange(n_samples):
		# Generate opd error
		exponent = np.random.uniform(low=min_power_law_exponent, high=max_power_law_exponent, size=None)
		if rms_log_space:
			input_rms = 10 ** np.random.uniform(low=np.log10(min_rms), high=np.log10(max_rms), size=None)
		else:
			input_rms = np.random.uniform(low=min_rms, high=max_rms, size=None)
		error = make_opd_error(exponent=exponent, rms=input_rms, aperture=aperture, pupil_diameter=pupil_diameter, remove_modes=ptt_b)
		
		input_opd_coeffs = np.matmul(Z_inv, error)
		rms = np.sqrt(np.sum(np.square(input_opd_coeffs)))
		input_opd_coeffs *= (input_rms / rms)

		input_opd_coeffs_set[k,:] = input_opd_coeffs

	return input_opd_coeffs_set

######################
# Monochromatic scalar ZWFS reconstructor
######################

# Cost function
def cost_function_monoc(x, Imeas, Z, aperture, zwfs, wavelength, nphot):

	# Forward
	wf_out = zwfs_forward_opd(x, Z, aperture, zwfs, wavelength, nphot)
	Iout_k = wf_out.power

	# Calculate cost function
	J = np.square(np.linalg.norm(Imeas - Iout_k, ord=2)) * 1e4 / (nphot**2)

	return J

# Calculate gradients based on backpropagation
def calc_grad_monoc(x, Imeas, Z, aperture, zwfs, wavelength, nphot):

	Ein_k = Wavefront(Field(aperture * np.exp(1j*(2*np.pi/wavelength)*np.matmul(Z,x*1e-9)), aperture.grid), wavelength) # E field at input corresponding to coefficients at x
	Ein_k.total_power = nphot

	wf_model = zwfs_forward_opd(x, Z, aperture, zwfs, wavelength, nphot) # Wavefront at output
	Eout_k = wf_model.electric_field # E field at output
	Ik = wf_model.power # Intensity at output

	dIk = -2 * (Imeas - Ik)
	dEout_k = 2 * dIk * Eout_k
	dEin_k = zwfs.backward(Wavefront(Field(dEout_k, Eout_k.grid), wavelength)).electric_field

	dphi_vector = np.matmul(Z.T, np.imag(dEin_k * np.conj(Ein_k.electric_field)))  * ((2*np.pi) / wavelength) * 1e-9

	correction_factor = 1 / (1.27e-4 * 2665.5 * (nphot**2)) # correction factor for certain setup, found by comparing to numerically calculated gradients
	dphi_vector = dphi_vector * correction_factor
	return dphi_vector

# Simulate reconstructed wavefront for a certain input error
def sim_recon_monoc(input_coeffs, aperture, Z, phase_step, phase_dot_diameter, pupil_diameter, wavelength, nphot, xtol=1e-2, maxiter=50, add_noise=False):
	
	# Zernike wavefront sensor model
	zwfs = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_step, phase_dot_diameter=phase_dot_diameter, num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=wavelength)
	
	# Zernike wavefront sensor response actual input
	wf_meas = zwfs_forward_opd(input_coeffs, Z, aperture, zwfs, wavelength, nphot)
	Imeas = wf_meas.power

	if add_noise == True:
		Imeas = large_poisson(Imeas, thresh=1e6)
	
	# Reconstruction
	x0 = np.zeros(np.shape(Z)[1])
	theta = minimize(cost_function_monoc, x0, args=(Imeas, Z, aperture, zwfs, wavelength, nphot), method='Newton-CG', jac=calc_grad_monoc, options={'maxiter':maxiter, 'disp':False, 'xtol':xtol})
	
	return theta

######################
# Multiwavelength scalar ZWFS reconstructor
######################

# Cost function
def cost_function_mw(x, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot):

	Js = np.zeros(len(wavelengths))

	# Forward
	for i, w in enumerate(wavelengths):
		wf_out = zwfs_forward_opd(x, Z, aperture, zwfs_models[i], w, nphot)
		Iout_k = wf_out.power

		# Calculate cost function
		J = np.square(np.linalg.norm(Is_meas[i] - Iout_k, ord=2)) * 1e4 / (nphot**2)
		Js[i] = J

	return np.sum(Js)

# Calculate gradients based on backpropagation
def calc_grad_mw(x, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot):

	dopd_vectors = np.zeros((len(wavelengths), len(x)))

	# calculate gradients at each wavelength
	for i, w in enumerate(wavelengths):
		Ein_k = Wavefront(Field(aperture * np.exp(1j*(2*np.pi/w)*np.matmul(Z,x*1e-9)), aperture.grid), w)
		Ein_k.total_power = nphot

		wf_model = zwfs_forward_opd(x, Z, aperture, zwfs_models[i], w, nphot) # Wavefront at output
		Eout_k = wf_model.electric_field # E field at output
		Ik = wf_model.power # Intensity at output

		dIk = -2 * (Is_meas[i] - Ik)
		dEout_k = 2 * dIk * Eout_k
		dEin_k = zwfs_models[i].backward(Wavefront(Field(dEout_k, aperture.grid), w)).electric_field

		Z_T = Z.T
		dopd_vector = np.matmul(Z_T, np.imag(dEin_k * np.conj(Ein_k.electric_field))) * ((2*np.pi) / w) * 1e-9

		correction_factor = 1 * len(wavelengths) / (1.27e-4 * 2665.5 * (nphot**2)) # correction factor for certain setup, found by comparing to numerically calculated gradients
		dopd_vector = dopd_vector * correction_factor

		dopd_vectors[i, :] = dopd_vector

	# take average of gradients over the wavelengths
	dopd_vector_average = np.average(dopd_vectors, axis=0)

	return dopd_vector_average

# Simulate reconstructed wavefront for a certain input error
def sim_recon_mwzwfs(input_coeffs, aperture, Z, design_phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=50, add_noise=True):

	# Zernike mask phase shifts and diameters for different wavelengths
	phase_steps = (design_wavelength / wavelengths) * design_phase_shift
	phase_dot_diameters = (design_wavelength / wavelengths) * design_diameter

	# Zernike wavefront sensor model each wavelength
	zwfs_models = []
	for i, w in enumerate(wavelengths):
		zwfs_i = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_steps[i], phase_dot_diameter=phase_dot_diameters[i], num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=w)
		zwfs_models.append(zwfs_i)
		
	# Zernike wavefront sensor response actual input
	Is_meas = aperture.grid.zeros((len(wavelengths),))
	for i, w in enumerate(wavelengths):
		wf_meas = zwfs_forward_opd(input_coeffs, Z, aperture, zwfs_models[i], w, nphot)
		Imeas = wf_meas.power
		Is_meas[i] = Imeas

	if add_noise == True:
		Is_meas = large_poisson(Is_meas, thresh=1e6)
		
	# Reconstruction
	x0 = np.zeros(np.shape(Z)[1])
	theta = minimize(cost_function_mw, x0, args=(Is_meas, Z, aperture, zwfs_models, wavelengths, nphot), method='Newton-CG', jac=calc_grad_mw, options={'maxiter':maxiter, 'disp':False, 'xtol':xtol})
	
	return theta

######################
# Monochromatic vZWFS reconstructor
######################

# Cost function
def cost_function_v(x, Is_meas, Z, aperture, zwfs_models, wavelength, nphot):

	Js = np.zeros(2)

	# Forward
	for i in np.arange(2):
		wf_out = zwfs_forward_opd(x, Z, aperture, zwfs_models[i], wavelength, nphot)
		Iout_k = wf_out.power

		# Calculate cost function
		J = np.square(np.linalg.norm(Is_meas[i] - Iout_k, ord=2)) * 1e4 / (nphot**2)
		Js[i] = J

	return np.sum(Js)

# Calculate gradients based on backpropagation
def calc_grad_v(x, Is_meas, Z, aperture, zwfs_models, wavelength, nphot):

	dopd_vectors = np.zeros((2, len(x)))

	# calculate gradients at each wavelength
	for i in np.arange(2):
		Ein_k = Wavefront(Field(aperture * np.exp(1j*(2*np.pi/wavelength)*np.matmul(Z,x*1e-9)), aperture.grid), wavelength)
		Ein_k.total_power = nphot
		
		wf_model = zwfs_forward_opd(x, Z, aperture, zwfs_models[i], wavelength, nphot) # Wavefront at output
		Eout_k = wf_model.electric_field # E field at output
		Ik = wf_model.power # Intensity at output

		dIk = -2 * (Is_meas[i] - Ik)
		dEout_k = 2 * dIk * Eout_k
		dEin_k = zwfs_models[i].backward(Wavefront(Field(dEout_k, aperture.grid), wavelength)).electric_field

		Z_T = Z.T 
		dopd_vector = np.matmul(Z_T, np.imag(dEin_k * np.conj(Ein_k.electric_field))) * ((2*np.pi) / wavelength) * 1e-9

		correction_factor = 1 / (1.27e-4 * 0.5 * 2665.5 * (nphot**2)) # correction factor for certain setup, found by comparing to numerically calculated gradients
		dopd_vector = dopd_vector * correction_factor

		dopd_vectors[i, :] = dopd_vector

	# take average of gradients over the wavelengths
	dopd_vector_average = np.average(dopd_vectors, axis=0)

	return dopd_vector_average

# Simulate reconstructed wavefront for a certain input error
def sim_recon_vzwfs(input_coeffs, aperture, Z, phase_step, phase_dot_diameter, pupil_diameter, wavelength, nphot, xtol=1e-2, maxiter=50, add_noise=True):

	# Zernike wavefront sensor model each wavelength
	zwfs_l = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_step, phase_dot_diameter=phase_dot_diameter, num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=wavelength)
	zwfs_r = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=-phase_step, phase_dot_diameter=phase_dot_diameter, num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=wavelength)

	zwfs_models = [zwfs_l, zwfs_r]
		
	# Zernike wavefront sensor response actual input
	Is_meas = aperture.grid.zeros((2,))
	for i in np.arange(2):
		wf_meas = zwfs_forward_opd(input_coeffs, Z, aperture, zwfs_models[i], wavelength, nphot)
		Imeas = wf_meas.power
		Is_meas[i] = Imeas

	if add_noise == True:
		Is_meas = large_poisson(Is_meas, thresh=1e6)
		
	# Reconstruction
	x0 = np.zeros(np.shape(Z)[1])
	theta = minimize(cost_function_v, x0, args=(Is_meas, Z, aperture, zwfs_models, wavelength, nphot), method='Newton-CG', jac=calc_grad_v, options={'maxiter':maxiter, 'disp':False, 'xtol':xtol})
	
	return theta

######################
# Multiwavelength vZWFS reconstructor
######################

# Cost function
def cost_function_mwv(x, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot):

	Js = np.zeros(2 * len(wavelengths))

	# Forward
	for j, w in enumerate(wavelengths):
		for i in np.arange(2):

			k = j*2 + i

			wf_out = zwfs_forward_opd(x, Z, aperture, zwfs_models[j][i], w, nphot)
			Iout_k = wf_out.power

			# Calculate cost function
			J = np.square(np.linalg.norm(Is_meas[k] - Iout_k, ord=2)) * 1e4 / (nphot**2) # make sure numbers aren't too small so scipy can optimize
			Js[k] = J

	return np.sum(Js)

# Calculate gradients based on backpropagation
def calc_grad_mwv(x, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot):

	dopd_vectors = np.zeros((2 * len(wavelengths), len(x)))

	# calculate gradients at each wavelength
	for j, w in enumerate(wavelengths):
		for i in np.arange(2):
			
			Ein_k = Wavefront(Field(aperture * np.exp(1j*(2*np.pi/w)*np.matmul(Z,x*1e-9)), aperture.grid), w)
			Ein_k.total_power = nphot

			wf_model = zwfs_forward_opd(x, Z, aperture, zwfs_models[j][i], w, nphot) # Wavefront at output
			Eout_k = wf_model.electric_field # E field at output
			Ik = wf_model.power # Intensity at output
			
			k = j*2 + i

			dIk = -2 * (Is_meas[k] - Ik)
			dEout_k = 2 * dIk * Eout_k
			dEin_k = zwfs_models[j][i].backward(Wavefront(Field(dEout_k, aperture.grid), w)).electric_field

			Z_T = Z.T 
			dopd_vector = np.matmul(Z_T, np.imag(dEin_k * np.conj(Ein_k.electric_field))) * ((2*np.pi) / w) * 1e-9

			correction_factor = 1 * len(wavelengths) / (1.27e-4 * 0.5 * 2665.5 * (nphot**2)) # correction factor for certain setup, found by comparing to numerically calculated gradients
			dopd_vector = dopd_vector * correction_factor

			dopd_vectors[k, :] = dopd_vector

	# take average of gradients over the wavelengths
	dopd_vector_average = np.average(dopd_vectors, axis=0)

	return dopd_vector_average

# Simulate reconstructed wavefront for a certain input error
def sim_recon_mwvzwfs(input_coeffs, aperture, Z, phase_step, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=50, add_noise=False):

	# Zernike mask phase shifts and diameters for different wavelengths
	phase_dot_diameters = (design_wavelength / wavelengths) * design_diameter

	zwfs_models = []
	for i, w in enumerate(wavelengths):
		zwfs_l = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_step, phase_dot_diameter=phase_dot_diameters[i], num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=w)
		zwfs_r = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=-phase_step, phase_dot_diameter=phase_dot_diameters[i], num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=w)

		zwfs_models.append([zwfs_l, zwfs_r])
		
	# Zernike wavefront sensor response actual input
	Is_meas = aperture.grid.zeros((2 * len(wavelengths),))
	for j, w in enumerate(wavelengths):
		for i in np.arange(2):
			wf_meas = zwfs_forward_opd(input_coeffs, Z, aperture, zwfs_models[j][i], w, nphot)
			Imeas = wf_meas.power
			k = j*2 + i
			Is_meas[k] = Imeas

	if add_noise == True:
		Is_meas = large_poisson(Is_meas, thresh=1e6)
		
	# Reconstruction
	x0 = np.zeros(np.shape(Z)[1])
	theta = minimize(cost_function_mwv, x0, args=(Is_meas, Z, aperture, zwfs_models, wavelengths, nphot), method='Newton-CG', jac=calc_grad_mwv, options={'maxiter':maxiter, 'disp':False, 'xtol':xtol})
	
	return theta

#########################
# Polychromatic
#########################

# Model broadband vZWFS response
def vzwfs_polych_model(input_coeffs, aperture, Z, phase_step, design_diameter, design_wavelength, pupil_diameter, wavelengths, nphot_i, add_noise=False):

	equiv_wavelength = np.average(wavelengths, weights=nphot_i)
	nphot_equiv = np.sum(nphot_i)

	phase_dot_diameter_eq = (design_wavelength / equiv_wavelength) * design_diameter
	
	# Zernike wavefront sensor model at equivalent wavelength
	zwfs_l = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_step, phase_dot_diameter=phase_dot_diameter_eq, num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=equiv_wavelength)
	zwfs_r = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=-phase_step, phase_dot_diameter=phase_dot_diameter_eq, num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=equiv_wavelength)

	zwfs_models = [zwfs_l, zwfs_r]
		
	# Zernike wavefront sensor response actual input
	Is_meas = aperture.grid.zeros((2,))
	for k in np.arange(2):
		for i, w in enumerate(wavelengths):

			if k == 0:
				phase_shift = phase_step
			else:
				phase_shift = -phase_step

			phase_dot_diameter = (design_wavelength / w) * design_diameter
			zwfs = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_shift, phase_dot_diameter=phase_dot_diameter, num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=w)

			wf_meas = zwfs_forward_opd(input_coeffs, Z, aperture, zwfs, w, nphot_i[i])
			Imeas = wf_meas.power
			Is_meas[k] += Imeas

	if add_noise == True:
		Is_meas = large_poisson(Is_meas, thresh=1e6)

	return zwfs_models, Is_meas, equiv_wavelength, nphot_equiv


#########################
# Phase unwrapping
#########################

# Make an ELT-like aperture (hexagonal segments width secondary spiders)
def make_elt_like_aperture(pupil_diameter, num_segment_rings, gap_size, spider_width, n_spiders, offset_spiders, with_spiders=True, return_segments=False):
	
	segment_size = (pupil_diameter - (2 * num_segment_rings + 1) * gap_size) / (2 * num_segment_rings + 1) # m

	elt_aperture_function, elt_segments = make_hexagonal_segmented_aperture(num_segment_rings, segment_size, gap_size, starting_ring=1, return_segments=True)

	spiders = [make_spider_infinite([0, 0], (360/n_spiders) * i + offset_spiders, spider_width) for i in range(n_spiders)]

	def elt_aperture_with_spiders(grid):
		aperture = elt_aperture_function(grid)

		if with_spiders:
			for spider in spiders:
				aperture *= spider(grid)

		return aperture

	if with_spiders and return_segments:
		# Use function to return the lambda, to avoid incorrect binding of variables
		def spider_func(grid):
			spider_aperture = grid.ones()
			for spider in spiders:
				spider_aperture *= spider(grid)
			return spider_aperture

		def segment_with_spider(segment):
			return lambda grid: segment(grid) * spider_func(grid)

		elt_segments = [segment_with_spider(s) for s in elt_segments]

	if return_segments:
		return elt_aperture_with_spiders, elt_segments
	else:
		return elt_aperture_with_spiders
	
# Electric field corresponding to mode coefficients
def coeffs_to_Efield(x, ref_seg_id, Z, aperture, wavelength):
	
	E_in = aperture * np.exp(1j*(2*np.pi/wavelength)*np.matmul(Z, x*1e-9))

	return E_in

# Mode coefficients to wavefront at ZWFS output
def zwfs_forward_segments(x, ref_seg_id, Z, aperture, zwfs, wavelength, nphot):

	E_in = coeffs_to_Efield(x, ref_seg_id, Z, aperture, wavelength)

	# HCIpy wavefront
	wf_in = Wavefront(Field(E_in, aperture.grid), wavelength)
	wf_in.total_power = nphot
	
	# Wavefront at output ZWFS
	wf_out = zwfs.forward(wf_in)

	return wf_out

# Model multiwavelength vZWFS. Return output image and propagation model for each pupil at each wavelength.
def mwvzwfs_model(input_coeffs, ref_seg_id, aperture, Z, phase_step, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, add_noise=False):
	# Zernike mask phase shifts and diameters for different wavelengths
	phase_dot_diameters = (design_wavelength / wavelengths) * design_diameter

	zwfs_models = []
	for i, w in enumerate(wavelengths):
		zwfs_l = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=phase_step, phase_dot_diameter=phase_dot_diameters[i], num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=w)
		zwfs_r = ZernikeWavefrontSensorOptics(input_grid=aperture.grid, phase_step=-phase_step, phase_dot_diameter=phase_dot_diameters[i], num_pix=64, pupil_diameter=pupil_diameter, reference_wavelength=w)

		zwfs_models.append([zwfs_l, zwfs_r])
		
	# Zernike wavefront sensor response actual input
	Is_meas = aperture.grid.zeros((2 * len(wavelengths),))
	for j, w in enumerate(wavelengths):
		for i in np.arange(2):
			wf_meas = zwfs_forward_segments(input_coeffs, ref_seg_id, Z, aperture, zwfs_models[j][i], w, nphot)
			Imeas = wf_meas.power
			k = j*2 + i
			Is_meas[k] = Imeas

	if add_noise == True:
		Is_meas = large_poisson(Is_meas, thresh=1e6)

	return Is_meas, zwfs_models

def cost_function_mwv_pet(x, ref_seg_id, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot):

	Js = np.zeros(2 * len(wavelengths))

	# Forward
	for j, w in enumerate(wavelengths):
		for i in np.arange(2):

			k = j*2 + i

			wf_out = zwfs_forward_segments(x, ref_seg_id, Z, aperture, zwfs_models[j][i], w, nphot)
			Iout_k = wf_out.power

			# Calculate cost function
			J = np.square(np.linalg.norm(Is_meas[k][aperture>0] - Iout_k[aperture>0], ord=2)) * 1e4 / (nphot**2)
			Js[k] = J

	return np.sum(Js)

def calc_grad_mwv_pet(x, ref_seg_id, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot, return_grads=False):

	dopd_vectors = np.zeros((2 * len(wavelengths), len(x)))

	# calculate gradients at each wavelength
	for j, w in enumerate(wavelengths):
		for i in np.arange(2):
			
			Ein_k = Wavefront(coeffs_to_Efield(x, ref_seg_id, Z, aperture, w), w)
			Ein_k.total_power = nphot

			wf_model = zwfs_forward_segments(x, ref_seg_id, Z, aperture, zwfs_models[j][i], w, nphot) # Wavefront at output
			Eout_k = wf_model.electric_field # E field at output
			Ik = wf_model.power # Intensity at output
			
			k = j*2 + i

			dIk = -2 * (Is_meas[k] - Ik)
			dEout_k = 2 * dIk * Eout_k
			dEin_k = zwfs_models[j][i].backward(Wavefront(Field(dEout_k, aperture.grid), w)).electric_field

			Z_T = Z.T
			dopd_vector = np.matmul(Z_T, np.imag(dEin_k * np.conj(Ein_k.electric_field))) * ((2*np.pi) / w) * 1e-9

			correction_factor = 1 * 4.68963606e-05 * 4 * len(wavelengths) / (1.27e-4 * (nphot**2)) # correction factor for certain setup, found by comparing to numerically calculated gradients
			dopd_vector = dopd_vector * correction_factor

			dopd_vectors[k, :] = dopd_vector
			
    # take average of gradients over the wavelengths
	dopd_vector_average = np.average(dopd_vectors, axis=0)

	if return_grads:
		return dopd_vectors
	else:
		return dopd_vector_average

# Simulate reconstructed wavefront for a certain input error
def sim_recon_mwvzwfs_dopd(input_coeffs, ref_seg_id, aperture, Z, phase_step, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=50, add_noise=False):

	Is_meas, zwfs_models = mwvzwfs_model(input_coeffs, ref_seg_id, aperture, Z, phase_step, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, add_noise=add_noise)
		
	# Reconstruction
	x0 = np.zeros(len(input_coeffs))
	theta = minimize(cost_function_mwv_pet, x0, args=(ref_seg_id, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot), method='Newton-CG', jac=calc_grad_mwv_pet, options={'maxiter':maxiter, 'disp':False, 'xtol':xtol})
	
	return theta

# Phase unwraps the wrapped phase response using two wavelengths
def phase_unwrap_twowavel_simple(wrapped_phases, wavelengths):
	ph1 = wrapped_phases[0].copy()
	ph2 = wrapped_phases[1].copy()

	m1 = ph1<0
	m2 = ph2<0

	# [-pi, 0] -> [pi, 2pi]
	ph1[m1] = ph1[m1] + (2*np.pi)
	ph2[m2] = ph2[m2] + (2*np.pi)

	# Phase unwrap
	coarse_map = (ph1 - ph2) % (np.pi*2)

	# [pi, 2pi] -> [-pi, 0]
	coarse_map[coarse_map>np.pi] -= (2*np.pi)

	# Calculate opd using equivalent wavelength
	w1 = wavelengths[0]
	w2 = wavelengths[1]
	equiv_l = (w1 * w2) / (np.abs(w1 - w2))
	opd = (equiv_l / (2*np.pi)) * coarse_map

	return opd

# Calculate the wrapped value of the opd
def wrapped_opd(x, wavelength):
	return (x-(wavelength/2))%wavelength - (wavelength/2)

# Simulate reconstructed petal error using a vZWFS and two-wavelength phase unwrapping
def sim_recon_vzwfs_pu(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter, xtol=1e-2, maxiter=50, add_noise=False):

	# Gradient descent on separate wavelengths
	th0 = sim_recon_mwvzwfs_dopd(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, np.array([wavelengths[0]]), nphot, pupil_diameter, add_noise=add_noise)
	th1 = sim_recon_mwvzwfs_dopd(petal_coeffs, ref_seg_id, aperture, Z, phase_shift, design_diameter, design_wavelength, np.array([wavelengths[1]]), nphot, pupil_diameter, add_noise=add_noise)

	w0_est_petal_coeffs = th0.x
	w1_est_petal_coeffs = th1.x

	# Different wavelengths can have different offset due to piston ambiguity --> take the first petal at first wavelength as reference
	reference_difference = w0_est_petal_coeffs[0] - w1_est_petal_coeffs[0]
	w1_est_petal_coeffs += reference_difference

	# Calculate the wrapped opd from the estimated opd
	w0_est_petal_coeffs = wrapped_opd(w0_est_petal_coeffs, wavelengths[0]*1e9)
	w1_est_petal_coeffs = wrapped_opd(w1_est_petal_coeffs, wavelengths[1]*1e9)

	w0_opd= Field(np.matmul(Z, w0_est_petal_coeffs*1e-9), aperture.grid)
	w1_opd = Field(np.matmul(Z, w1_est_petal_coeffs*1e-9), aperture.grid)

	# Calculated wrapped phase at each wavelength [-pi, pi]
	w0_wf = Wavefront(aperture * np.exp(1j * w0_opd * (2*np.pi/wavelengths[0])), wavelengths[0])
	w1_wf = Wavefront(aperture * np.exp(1j * w1_opd * (2*np.pi/wavelengths[1])), wavelengths[1])

	wrapped_phases = aperture.grid.zeros((2,))
	wrapped_phases[0] = w0_wf.phase
	wrapped_phases[1] = w1_wf.phase

	# Unwrap
	unwrapped_opd = phase_unwrap_twowavel_simple(wrapped_phases, wavelengths)

	# Estimated petal coefficients
	est_coeffs = np.matmul(inverse_tikhonov(Z), unwrapped_opd*1e9)

	return est_coeffs, w0_est_petal_coeffs, w1_est_petal_coeffs
