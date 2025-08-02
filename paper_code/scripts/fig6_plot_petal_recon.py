##############################
# Reproduces Figure 6 on petal reconstruction dynamic range
##############################

import matplotlib.pyplot as plt
import numpy as np

# Load data

recon_petal_coeffs = np.load('data/phase_unwrapping/600nm_700nm_phasestep1-57_dotdia2-00_petal_recons_coeffs_vzwfs_gd.npy')
recon_petal_coeffs_vzwfs_m = np.load('data/phase_unwrapping/700nm_phasestep1-57_dotdia2-00_recon_petal_coeffs_vzwfs_m.npy')
opd_amplitudes = np.linspace(-2500, 2500, num=40)

# Plot Parameters

plt.rcParams.update({'font.size': 16})
blue = '#4477AA'
red = '#EE6677'

# Plot figure

petal_id = 1
plt.figure(figsize=(10,8))

plt.plot(opd_amplitudes, recon_petal_coeffs_vzwfs_m[:, 1], marker='*', markersize=9, c=red, zorder=2, label='Mono-GD (700 nm)')
plt.plot(opd_amplitudes, recon_petal_coeffs[:, petal_id], marker='*', markersize=9, c=blue, zorder=1, label='Phase unwrapped (600, 700 nm)')
plt.xlabel('Petal OPD [nm]'), plt.ylabel('Reconstructed petal OPD [nm]'), plt.title('Single petal reconstruction'), plt.legend()

plt.show()