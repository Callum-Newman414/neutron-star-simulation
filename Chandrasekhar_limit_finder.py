import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8})
import numpy as np
from numpy import pi
from numpy import log10

from rk4_solver_functions import ode_solve_rk
from rk4_solver_functions import get_density

# Plotting radius as a function of mass #################################################################

r = np.arange(5e4, 5.005e7,5e4)

dr = r[2] - r[1]

# Here, the radius is estimated to be the distance from the centre where the density falls to
# 1/1000th the central value.
# The mass is taken as the mass at tjis value of r as well.

# Non-relativistic

rhoc = np.arange(6, 14.1, 0.1) # use 10**rhoc
radiuss_non = np.zeros(rhoc.size) # tracks the radius for each critical density being tested
masses_non = np.zeros(rhoc.size) # tracks the masses for different critical densities

for i in range(rhoc.size):
    (rho_i, mass_i) = get_density(10**(rhoc[i]), r, 0) # get the each mass and density for all r's
    radial_index = (np.nonzero(rho_i))[0].size
    masses_non[i] = mass_i[radial_index - 1]
    radiuss_non[i] = r[radial_index] # find how many non-zero elements there are

# Relativistic:

radiuss_rel = np.zeros(rhoc.size)
masses_rel = np.zeros(rhoc.size)

for i in range(rhoc.size):
    (rho_i, mass_i) = get_density(10**(rhoc[i]), r, 1) # get the each mass and density for all r's
    radial_index = (np.nonzero(rho_i))[0].size
    masses_rel[i] = mass_i[radial_index - 1]
    radiuss_rel[i] = r[radial_index] # find how many non-zero elements there are

l = np.nonzero(masses_rel)[0].size - 1
print(l)
fig3, axs3 = plt.subplots()
axs3.plot(masses_non[:l], radiuss_non[:l], color = 'blue', label='Non-relativistic')
axs3.plot(masses_rel, radiuss_rel, color='red', label='Relativistic')
axs3.set_ylabel('Radius of star /m')
axs3.set_xlabel('Mass of star /kg')
axs3.set_title('Radius of a white dwarf star against its mass')
axs3.set_xlim(0, 5e30)
axs3.set_ylim(0, 3.5e7)
# Draw line showing the Chandrasekhar limit:
axs3.axvline(x = masses_rel.max(), label='Chandrasekhar mass', color='purple')
axs3.legend()

plt.tight_layout()
print(masses_rel[-1])
plt.show()
