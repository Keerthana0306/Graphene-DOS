import numpy as np
import matplotlib.pyplot as plt
a = 2.46 
t = 2.7  
n_points = 1000 
energy_resolution = 0.05 
kx_vals = np.linspace(-np.pi / a, np.pi / a, n_points)
ky_vals = np.linspace(-np.pi / a, np.pi / a, n_points)
kx, ky = np.meshgrid(kx_vals, ky_vals)
E = lambda kx, ky: t * np.sqrt(1 + 4 * np.cos(np.sqrt(3) * kx * a / 2) * np.cos(ky * a / 2) +
                               4 * np.cos(ky * a / 2)**2)
E_k = E(kx, ky)

E_flat = E_k.flatten()
E_min, E_max = -3 * t, 3 * t
E_bins = np.arange(E_min, E_max, energy_resolution)
DOS, bin_edges = np.histogram(E_flat, bins=E_bins, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.figure(figsize=(8, 6))
plt.plot(bin_centers, DOS, color="purple", lw=2)
plt.title("DOS for Graphene", fontsize=14)
plt.xlabel("eV", fontsize=12)
plt.ylabel("DOS (States/eV/Unit Cell)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()