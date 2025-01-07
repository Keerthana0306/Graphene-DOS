import numpy as np
import matplotlib.pyplot as plt
t = 1 
a = 1  
k_points = 200
kx = np.linspace(-np.pi / a, np.pi / a, k_points)
ky = np.linspace(-np.pi / a, np.pi / a, k_points)
kx, ky = np.meshgrid(kx, ky)
E = -2 * t * (np.cos(kx * a) + np.cos(ky * a))
E_flat = E.flatten()
bins = 100  
dos, energy_bins = np.histogram(E_flat, bins=bins, density=True)
energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].set_title('Energy Dispersion')
contour = ax[0].contourf(kx, ky, E, levels=50, cmap='viridis')
plt.colorbar(contour, ax=ax[0], label='Energy (E)')
ax[0].set_xlabel(r'$k_x$')
ax[0].set_ylabel(r'$k_y$')
ax[1].set_title('Density of States (DOS)')
ax[1].plot(energy_centers, dos, color='red', label='DOS')
ax[1].set_xlabel('Energy (E)')
ax[1].set_ylabel('Density of States')
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].legend()
plt.tight_layout()
plt.show()
