##############################
# Reproduces aperture inset on Figure 6
##############################

import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from utils import make_elt_like_aperture

# Optical system
pupil_diameter = 1 # [m]
focal_length = 1 # [m]

# Detector
num_pix = 128

# Pupil
pupil_grid = make_pupil_grid(dims=num_pix, diameter=pupil_diameter*1.1)

num_segment_rings = 3
gap_size = 8e-3
spider_width = 0.02
n_spiders = 6
offset_spiders = 0
eltapg, segments = make_elt_like_aperture(pupil_diameter=pupil_diameter, num_segment_rings=num_segment_rings, gap_size=gap_size, spider_width=spider_width, n_spiders=n_spiders, offset_spiders=offset_spiders, with_spiders=True, return_segments=True)

# Aperture
aperture = evaluate_supersampled(eltapg, pupil_grid, 1)

petal_id = 1

petal_coeffs = np.zeros(n_spiders)
petal_coeffs[petal_id] = 150

angle = Field(np.arctan2(aperture.grid.y, aperture.grid.x), pupil_grid)
angle[angle<0] += 2*np.pi
angle = np.rad2deg(angle)


spider_angles = (360/n_spiders) * np.arange(n_spiders) + offset_spiders
spider_angles = np.append(spider_angles, 360)

petal_tf = aperture.grid.zeros((n_spiders,))
for i in np.arange(n_spiders):
	
	petal = (angle > spider_angles[i]) * (angle <= spider_angles[(i+1)%(n_spiders+1)]) * aperture

	petal_tf[i] = petal

petal_basis = ModeBasis(petal_tf.T)

plt.figure()
piston = np.average(Field(np.matmul(petal_basis.transformation_matrix, petal_coeffs), pupil_grid)[aperture>0])
f = Field(np.matmul(petal_basis.transformation_matrix, petal_coeffs), pupil_grid)
f[aperture>0] -= piston
imshow_field(f, cmap='RdBu', vmin=-120, vmax=120), plt.title('Petal error [nm]'), plt.colorbar(), plt.xticks([]), plt.yticks([])
plt.show()