##############################
# Reproduces Figure 5 on bandwidth
##############################

import matplotlib.pyplot as plt
import numpy as np

# Load data 

rms_error_vzwfs = np.load('data/polychromatic/rms_error_vzwfs_l0600_nlambda10_nphotnm100_50samples_10bwsamples.npy')
rms_error_mwvzwfs = np.load('data/polychromatic/rms_error_mwvzwfs_l0600_nlambda10_nphotnm100_50sample_10bwsampless.npy')

input_rms = 20
bw = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) / 100 # %
center_l = 600e-9
nphotnm = 100

# Plot parameters

plt.rcParams.update({'font.size': 16})

blue = '#4477AA'
red = '#EE6677'
purple= '#AA3377'

# Plot figure

fig, ax = plt.subplots(figsize=(8.5,7.5))

plt.title(f'Input {input_rms} nm RMS')

yest_vzwfs = np.average(rms_error_vzwfs, axis=1)
yerr_vzwfs = np.sqrt(np.var(rms_error_vzwfs, axis=1))
plt.plot(bw*100, yest_vzwfs, c=blue, label='vZWFS')
plt.fill_between(bw*100, yest_vzwfs - yerr_vzwfs, yest_vzwfs + yerr_vzwfs, color=blue, alpha=0.5)

yest_mwvzwfs = np.average(rms_error_mwvzwfs, axis=1)
yerr_mwvzwfs = np.sqrt(np.var(rms_error_mwvzwfs, axis=1))
plt.plot(bw*100, yest_mwvzwfs, c=red, label='mw-vZWFS')
plt.fill_between(bw*100, yest_mwvzwfs - yerr_mwvzwfs, yest_mwvzwfs + yerr_mwvzwfs, color=red, alpha=0.5)

deltal = np.linspace(10, 100)
pn_scale = 1 / np.sqrt(deltal) * 17 # Match factor to curve
bw_scale = np.square(deltal) / 2300 # Match factor to curve
plt.plot(deltal, pn_scale, c='black', linestyle='dotted', label='$\\propto$  $1/\\sqrt{\\Delta\\lambda}$')
plt.plot(deltal, bw_scale, c='black', linestyle='dashed', label='$\\propto$  $\\Delta\\lambda^2$')

plt.legend(), plt.xlabel('Bandwidth [$\Delta\lambda/\lambda_{0} \\times 100$]'), plt.ylabel('Residual RMS [nm]')

nphot_tot = nphotnm * center_l * 1e9 * bw
ax2 = ax.twiny()
ax2.plot(nphot_tot, np.ones(len(bw)), alpha=0)
ax2.set_xlabel("$N_{phot,tot}$")

plt.show()