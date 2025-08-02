##############################
# Reproduces Figure 3 on convergence failure
##############################

import matplotlib.pyplot as plt
from hcipy import *
import numpy as np
from utils import *

# Load data

input_opd_coeffs_set = np.load('data/dynamic_range_simulation/input_opd_coeffs_set.npy')
aperture = read_field('data/dynamic_range_simulation/aperture.pkl')
pupil_grid = aperture.grid
Z = np.load('data/dynamic_range_simulation/Z.npy')
num_modes = np.shape(Z)[1]

est_coeffs_mwzwfs = np.load('data/dynamic_range_simulation/600nm_1000nm_phasestep7-85_dotdia2-00_est_coeffs_mwzwfs.npy')

# Optical system

pupil_diameter = 1
wavelengths = np.array([600e-9, 1000e-9])
design_phase_shift = 5*np.pi/2
design_diameter = 2
design_wavelength = wavelengths[0]

nphot=1e7

# Failed reconstruction index
i = 177

# Model multi-wavelength ZWFS. Input Zernike coefficients, output intensity images and propagation models for each wavelength
def mwzwfs_model(input_coeffs, aperture, Z, design_phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter):
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

	return Is_meas, zwfs_models

# Calculate gradient at certain coefficient vector x. Returns gradient calculated at each wavelength and the overall average
def calc_grad_mw_test(x, Is_meas, Z, aperture, zwfs_models, wavelengths, nphot):

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

	return dopd_vectors, dopd_vector_average 

Is_meas, zwfs_models = mwzwfs_model(input_opd_coeffs_set[i], aperture, Z, design_phase_shift, design_diameter, design_wavelength, wavelengths, nphot, pupil_diameter)

grads_init, grads_init_avg = calc_grad_mw_test(np.zeros(num_modes), Is_meas, Z, aperture, zwfs_models, wavelengths, nphot)
grads_conv, grads_conv_avg = calc_grad_mw_test(est_coeffs_mwzwfs[i], Is_meas, Z, aperture, zwfs_models, wavelengths, nphot)
grads_true, grads_true_avg = calc_grad_mw_test(input_opd_coeffs_set[i], Is_meas, Z, aperture, zwfs_models, wavelengths, nphot)

# Plot

blue = '#4477AA'
red = '#EE6677'
purple= '#AA3377'
green = '#228833'

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [0.08, 0.08, 0.11],})
fig.tight_layout()
fig.set_size_inches(20, 5)

opd_input = Field(np.matmul(Z,input_opd_coeffs_set[i,:]), pupil_grid)
vmin = -np.max([abs(np.min(opd_input)), abs(np.max(opd_input))])
vmax = np.max([abs(np.min(opd_input)), abs(np.max(opd_input))])
imshow_field(opd_input, cmap='RdBu', vmin=vmin, vmax=vmax, ax=ax[0], aspect='auto'), plt.colorbar(), ax[0].set_xticks([], []), ax[0].set_yticks([], []), ax[0].set_title('(a) Input OPD error [nm]')

fd = Field(np.matmul(Z,input_opd_coeffs_set[i,:] - est_coeffs_mwzwfs[i,:]) , pupil_grid)
imshow_field(fd, cmap='RdBu', vmin=np.min(fd[aperture>0]), vmax=-np.min(fd[aperture>0]), ax=ax[1], aspect='auto'), plt.colorbar(), ax[1].set_xticks([], []), ax[1].set_yticks([], []), ax[1].set_title('(b) Residual [nm]')

start_mode = 4
ax[2].plot(np.arange(num_modes) + start_mode, grads_conv[0]/np.max(grads_conv[0]), label='$\\lambda_{0}$', c=blue)
ax[2].plot(np.arange(num_modes) + start_mode, grads_conv[1]/np.max(grads_conv[0]), label='$\\lambda_{1}$', c=red)
ax[2].plot(np.arange(num_modes) + start_mode, grads_conv_avg, linestyle='dotted', label='$Average$', c=green), ax[2].set_title('(c) Gradients at convergence'), ax[2].set_xlabel('Zernike mode'), ax[2].set_ylabel('Gradient [arbitrary unit]'), ax[2].set_ylim([-1.5, 1.5])
ax[2].legend()

fig.tight_layout()

plt.show()