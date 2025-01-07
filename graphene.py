
import numpy as np
import matplotlib.pyplot as plt
a = 1.42 
t = 2.8   
kx_vals = np.linspace(-np.pi / a, np.pi / a, 300)
ky_vals = np.linspace(-np.pi / a, np.pi / a, 300)
kx, ky = np.meshgrid(kx_vals, ky_vals)

E1 = t * np.sqrt(1 + 4 * np.cos(np.sqrt(3) * ky * a / 2) * np.cos(3 * kx * a / 2) +
                 4 * np.cos(np.sqrt(3) * ky * a / 2)**2)
E2 = -E1
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kx, ky, E1, cmap='viridis',alpha=0.8)
ax.plot_surface(kx, ky, E2, cmap='plasma', alpha=0.8)
ax.set_xlabel(r"$k_x$ (1/$\mathrm{\AA}$)", labelpad=10)
ax.set_ylabel(r"$k_y$ (1/$\mathrm{\AA}$)", labelpad=10)
ax.set_zlabel("Energy (eV)", labelpad=10)
ax.set_title("Graphene Band Structure", fontsize=14)
plt.show()
