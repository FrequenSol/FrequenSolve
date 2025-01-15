import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def von_karman_spectral_density(kx, ky, k0_list, nu, ax = 1.0, ay = 1.0, C=1.0):
   """
   Computes the von Kármán spectral density for given wavenumbers.

   Parameters:
      kx, ky: 2D arrays of wavenumbers in x and y directions.
      k0: Correlation wavenumber (characteristic scale).
      nu: Smoothness parameter.
      C: Scaling constant (default is 1.0).

   Returns:
      2D array of spectral density values.
   """
   k = np.sqrt((ax * kx)**2 + (ay * ky)**2)
   spectral_density = np.zeros(np.shape(k))
   for k0 in k0_list:
      spectral_density += C / (1 + (k / k0)**2)**(nu + 1)
   
   return spectral_density
    

def generate_stochastic_field(n, L, k0_list, nu, aniso, seed=None):
    """
    Generates a stochastic field using the von Kármán spectral density.

    Parameters:
        n: Grid size (number of points in each dimension).
        L: Physical size of the domain.
        k0: Correlation wavenumber (characteristic scale).
        nu: Smoothness parameter.
        seed: Random seed for reproducibility (optional).

    Returns:
        2D array representing the stochastic field.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Frequency domain grid
    kx = np.fft.fftfreq(n[0], d=L[0]/n[0]) * 2 * np.pi
    ky = np.fft.fftfreq(n[1], d=L[1]/n[1]) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    
    # Spectral density
    psd = von_karman_spectral_density(kx, ky, k0_list, nu, ax = aniso[0], ay = aniso[1])
    
    # Generate random Fourier coefficients
    random_phase = np.exp(2j * np.pi * np.random.random((n[1], n[0])))
    amplitude = np.sqrt(psd)
    fft_field = amplitude * random_phase
    
    # Transform to spatial domain
    field = np.fft.ifft2(fft_field).real
    
    field = (field - np.average(field))
    field *= 0.5 / np.max(field)
    field += 1
    return field
    
def get_interpolant(x,y):
   sorted_points = sorted(zip(x, y))
   sorted_x, sorted_y = zip(*sorted_points)
   
   return interp1d(sorted_x, sorted_y, kind='linear')

## Parameters
#n       = [1344,   300]
#L       = [4.029, 0.36]
#k0_list = [1, 10]                # Correlation wavenumber
#nu      = 0.1                    # Smoothness parameter
#aniso   = [1.0, 1.0]
#
#seed = 42                   # Fix seed
#
## Generate field
#perturb = generate_stochastic_field(n, L, k0_list, nu, aniso, seed)
#
#
#Vp0 = 15.1; k = 450
#x = np.linspace(0,L[0],n[0])
#z = np.linspace(10,1000*L[1],n[1])
#
#topo = np.fromfile(f"examples/desert_2d/data/dunes@", dtype = np.float32)
#surf = get_interpolant(np.linspace(0,L[0],len(topo)),topo)
#topo = surf(x)
#
#_, z = np.meshgrid(x,z)
#z -= topo * 1e3 + L[1] * 500
#z = np.maximum(5,10*z)
#
#Vp  = 1e-3 * Vp0 * np.sqrt(1 + k * z ** (1/3))
#Vs  = Vp / 2.5
#
#Rho0 = 840; k = 1
#Rho = 1e-3 * Rho0 * np.sqrt(1 + k * z ** (0.2))
#
#Vp  = np.single(Vp  * perturb)
#Vs  = np.single(Vs  * perturb)
#Rho = np.single(Rho * perturb)
#
#Vp.tofile( f"../fresh_hp3d/trunk/problems/SEISMIC/examples/desert_2d/data/Vp_dunes" )
#Vs.tofile( f"../fresh_hp3d/trunk/problems/SEISMIC/examples/desert_2d/data/Vs_dunes" )
#Rho.tofile(f"../fresh_hp3d/trunk/problems/SEISMIC/examples/desert_2d/data/Rho_dunes")
#
## Plot the field
#plt.figure(figsize=(8, 6))
#plt.imshow(Vp,
#           extent = (0, L[0], 0, L[1]),
#           cmap   = 'viridis')
#plt.colorbar(label='Amplitude')
#plt.show()
#
