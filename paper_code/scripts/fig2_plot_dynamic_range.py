##############################
# Reproduces Figure 2 on dynamic range
##############################

from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# Load data

input_opd_coeffs_set = np.load('data/dynamic_range_simulation/input_opd_coeffs_set.npy') # Input coefficients
aperture = read_field('data/dynamic_range_simulation/aperture.pkl') # Aperture
Z = np.load('data/dynamic_range_simulation/Z.npy') # Transformation matrix

est_coeffs_monoc_600 = np.load('data/dynamic_range_simulation/600nm_phasestep1-57_dotdia2-00_est_coeffs_monoc.npy')
est_coeffs_monoc_1000 = np.load('data/dynamic_range_simulation/1000nm_phasestep1-57_dotdia1-20_est_coeffs_monoc.npy')
est_coeffs_mwzwfs = np.load('data/dynamic_range_simulation/600nm_1000nm_phasestep7-85_dotdia2-00_est_coeffs_mwzwfs.npy')
est_coeffs_vzwfs_600 = np.load('data/dynamic_range_simulation/600nm_phasestep1-57_dotdia2-00_est_coeffs_vzwfs.npy')
est_coeffs_vzwfs_1000 = np.load('data/dynamic_range_simulation/1000nm_phasestep1-57_dotdia1-20_est_coeffs_vzwfs.npy')
est_coeffs_mwvzwfs = np.load('data/dynamic_range_simulation/600nm_1000nm_phasestep1-57_dotdia2-00_est_coeffs_mwvzwfs.npy')

# Plotting parameters

plt.rcParams.update({'font.size': 16})
xlabel = 'Input RMS [nm]'
ylabel = 'Residual RMS [nm]'
xlim = [-10, 250]
ylim = [-10, 250]
rose = '#CC6677'
blue = '#4477AA'
grey = '#BBBBBB'
wine = '#882255'

# RMS analysis

def rms_analysis(input_opd_coeffs_set, est_opd_coeffs_set, Z, aperture):

	pupil_grid = aperture.grid

	n_samples = np.shape(input_opd_coeffs_set)[0]
	
	input_rms_set = np.zeros(n_samples)
	est_rms_set = np.zeros(n_samples)
	diff = np.zeros(n_samples)

	for k in np.arange(n_samples):
		opd_in = Field(np.matmul(Z, input_opd_coeffs_set[k,:]), pupil_grid)
		opd_est = Field(np.matmul(Z, est_opd_coeffs_set[k,:]), pupil_grid)
		difference = opd_in - opd_est

		rms_in = np.sqrt(np.average(np.square(opd_in[aperture>0])))
		rms_est = np.sqrt(np.average(np.square(opd_est[aperture>0])))
		rms_diff = np.sqrt(np.average(np.square(difference[aperture>0])))

		input_rms_set[k] = rms_in
		est_rms_set[k] = rms_est
		diff[k] = rms_diff

	return input_rms_set, est_rms_set, diff


input_rms_set, est_rms_monoc_600_set , diff_monoc_600 =	rms_analysis(input_opd_coeffs_set, est_coeffs_monoc_600, Z, aperture)

input_rms_set, est_rms_monoc_1000_set , diff_monoc_1000 = rms_analysis(input_opd_coeffs_set, est_coeffs_monoc_1000, Z, aperture)

input_rms_set, est_rms_mwzwfs_set , diff_mwzwfs = rms_analysis(input_opd_coeffs_set, est_coeffs_mwzwfs, Z, aperture)

input_rms_set, est_rms_vzwfs_600_set , diff_vzwfs_600 =	rms_analysis(input_opd_coeffs_set, est_coeffs_vzwfs_600, Z, aperture)

input_rms_set, est_rms_vzwfs_1000_set , diff_vzwfs_1000 = rms_analysis(input_opd_coeffs_set, est_coeffs_vzwfs_1000, Z, aperture)

input_rms_set, est_rms_mwvzwfs_set , diff_mwvzwfs =	rms_analysis(input_opd_coeffs_set, est_coeffs_mwvzwfs, Z, aperture)

# Plot

s = 100
text_start = 230
text_spacing = 15

fig = plt.figure(figsize=(22,13))

fig.supxlabel('Input RMS [nm]')
fig.supylabel('Residual RMS [nm]')

plt.subplot(231)
plt.title('(a)')
plt.plot([0, 250], [0, 250], c=grey, linestyle='dashed')
plt.scatter(input_rms_set, diff_monoc_600, c=blue, alpha=0.85, marker=".", s=s)
plt.xlim(-10, 250)
plt.ylim(-10, 250)
plt.xticks([])
plt.text(0, text_start, 'Scalar mask')
plt.text(0, text_start - text_spacing, '$\\lambda_{0}: 600$ nm')
plt.text(0, text_start - 2 * text_spacing, 'phase shift: $\\pi/2$')
plt.text(0, text_start - 3 * text_spacing, 'dot size: 2 $\\lambda/D$')

plt.subplot(232)
plt.title('(b)')
plt.plot([0, 250], [0, 250], c=grey, linestyle='dashed')
plt.scatter(input_rms_set, diff_monoc_1000, c=blue, alpha=0.85, marker=".", s=s)
plt.xlim(-10, 250)
plt.ylim(-10, 250)
plt.xticks([])
plt.yticks([])
plt.text(0, text_start, 'Scalar mask')
plt.text(0, text_start - text_spacing, '$\\lambda_{0}: 1000$ nm')
plt.text(0, text_start - 2 * text_spacing, 'phase shift: $\\pi/2$')
plt.text(0, text_start - 3 * text_spacing, 'dot size: 1.2 $\\lambda/D$')

plt.subplot(233)
plt.title('(c)')
plt.plot([0, 250], [0, 250], c=grey, linestyle='dashed')
plt.scatter(input_rms_set, diff_mwzwfs, c=blue, alpha=0.85, marker=".", s=s)
plt.xlim(-10, 250)
plt.ylim(-10, 250)
plt.xticks([])
plt.yticks([])
plt.text(0, text_start, 'Scalar mask')
plt.text(0, text_start - 1 * text_spacing, '$\\lambda_{0}: 600$ nm')
plt.text(0, text_start - 2 * text_spacing, '$\\lambda_{1}: 1000$ nm')
plt.text(0, text_start - 3 * text_spacing, 'phase shift: $5\\pi/2$ @ $\\lambda_{0}$')
plt.text(0, text_start - 4 * text_spacing, 'dot size: 2 $\\lambda/D$ @ $\\lambda_{0}$')

plt.subplot(234)
plt.title('(d)')
plt.plot([0, 250], [0, 250], c=grey, linestyle='dashed')
plt.scatter(input_rms_set, diff_vzwfs_600, c=blue, alpha=0.85, marker=".", s=s)
plt.xlim(-10, 250)
plt.ylim(-10, 250)
plt.text(0, text_start, 'Vector mask')
plt.text(0, text_start - 1 * text_spacing, '$\\lambda_{0}: 600$ nm')
plt.text(0, text_start - 2 * text_spacing, 'phase shift: $\\pm\\pi/2$')
plt.text(0, text_start - 3 * text_spacing, 'dot size: 2 $\\lambda/D$')

plt.subplot(235)
plt.title('(e)')
plt.plot([0, 250], [0, 250], c=grey, linestyle='dashed')
plt.scatter(input_rms_set, diff_vzwfs_1000, c=blue, alpha=0.85, marker=".", s=s)
plt.xlim(-10, 250)
plt.ylim(-10, 250)
plt.yticks([])
plt.text(0, text_start, 'Vector mask')
plt.text(0, text_start - 1 * text_spacing, '$\\lambda_{0}: 1000$ nm')
plt.text(0, text_start - 2 * text_spacing, 'phase shift: $\\pm\\pi/2$')
plt.text(0, text_start - 3 * text_spacing, 'dot size: 1.2 $\\lambda/D$')

plt.subplot(236)
plt.title('(f)')
plt.plot([0, 250], [0, 250], c=grey, linestyle='dashed')
plt.scatter(input_rms_set, diff_mwvzwfs, c=blue, alpha=0.85, marker=".", s=s)
plt.xlim(-10, 250)
plt.ylim(-10, 250)
plt.yticks([])
plt.text(0, text_start, 'Vector mask')
plt.text(0, text_start - 1 * text_spacing, '$\\lambda_{0}: 600$ nm')
plt.text(0, text_start - 2 * text_spacing, '$\\lambda_{1}: 1000$ nm')
plt.text(0, text_start - 3 * text_spacing, 'phase shift: $\\pm\\pi/2$')
plt.text(0, text_start - 4 * text_spacing, 'dot size: 2 $\\lambda/D$ @ $\\lambda_{0}$')

fig.tight_layout(pad=1.2, h_pad=0.7, w_pad=0.4)

plt.show()