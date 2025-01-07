import numpy as np
import matplotlib.pyplot as plt
t = 1  
a = 1  
k_points = 100
kx = np.linspace(4*(-np.pi) / a, 4*(np.pi) / a, k_points)
ky = 0
E = -2 * t * (np.cos(kx * a) + np.cos(ky * a))
plt.figure(figsize=(8, 5))
plt.plot(kx, E, label=r'$E(k_x, k_y=0)$', color='blue')
plt.xlabel(r'$k_x$')
plt.ylabel('energy')
plt.title('dispersion')
plt.axhline(0, color='red', linewidth=0.5, linestyle='--') 
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
