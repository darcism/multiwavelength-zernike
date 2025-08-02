##############################
# Reproduces Figure 4 on photon noise sensitivty
##############################

import matplotlib.pyplot as plt
import numpy as np

# Load data 

rms_errors_zwfs_600 = np.load('data/noise_analysis/rms_errors_zwfs_600_2ld_pi2_lowphot.npy')
rms_errors_mwzwfs_600_1000 = np.load('data/noise_analysis/rms_errors_mwzwfs_600_1000_2ld_5pi2_lowphot.npy')
rms_errors_mwzwfs_600_428 = np.load('data/noise_analysis/rms_errors_mwzwfs_600_428_2ld_5pi2_lowphot.npy')

nphot_array = np.array([5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])

# Plot

plt.rcParams.update({'font.size': 16})

blue = '#4477AA'
red = '#EE6677'
purple= '#AA3377'
green = '#228833'

plt.figure(figsize=(7.5,6.5))

yest_zwfs = np.average(rms_errors_zwfs_600, axis=1)
yerr_zwfs = np.sqrt(np.var(rms_errors_zwfs_600, axis=1))
plt.plot(nphot_array, yest_zwfs, c=blue, label='ZWFS (600 nm)')
plt.fill_between(nphot_array, yest_zwfs - yerr_zwfs, yest_zwfs + yerr_zwfs, color=blue, alpha=0.3)

yest_mwzwfs = np.average(rms_errors_mwzwfs_600_1000, axis=1)
yerr_mwzwfs = np.sqrt(np.var(rms_errors_mwzwfs_600_1000, axis=1))
plt.plot(nphot_array, yest_mwzwfs, c=red, label='mw-ZWFS (600, 1000 nm)')
plt.fill_between(nphot_array, yest_mwzwfs - yerr_mwzwfs, yest_mwzwfs + yerr_mwzwfs, color=red, alpha=0.3)

yest_mwzwfs = np.average(rms_errors_mwzwfs_600_428, axis=1)
yerr_mwzwfs = np.sqrt(np.var(rms_errors_mwzwfs_600_428, axis=1))
plt.plot(nphot_array, yest_mwzwfs, c=green, label='mw-ZWFS (428, 600 nm)')
plt.fill_between(nphot_array, yest_mwzwfs - yerr_mwzwfs, yest_mwzwfs + yerr_mwzwfs, color=green, alpha=0.3)

plt.title('Input 20 nm RMS'), plt.legend(), plt.xscale('linear'), plt.yscale('linear'), plt.xlabel('$N_{phot}$ per wavelength'), plt.ylabel('Residual RMS [nm]'), plt.xlim([5000, 80000])
plt.xticks(nphot_array[::2])

plt.show()