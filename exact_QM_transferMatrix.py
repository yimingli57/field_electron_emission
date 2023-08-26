#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:22:32 2023

@author: Yiming Li

Find the emission currents using the exact quantum mechanical method
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from Desktop.MNED.Code.WKB_emission import trans_coef_WKB

# Conversion constants
Hartree_to_eV = 27.2114
Bohr_to_m = 5.29177e-11

# Physical constants
m_e = 9.10938e-31 # Electron mass in kg
rho_e = 1.60218e-19 # Elementary charge in C
h_bar = 1.05457e-34 # Reduced Planck constant in J*s
epsilon_0 = 8.85419e-12 # Vacuum permittivity in F/m

# Measurable constants
phi = 5.47; # work function for Au(100) in eV
n = 19.3e3 / (196.97 * 1.67377e-27) # Density of free electrons in Au
E_F = h_bar**2 / (2 * m_e) * (3 * np.pi**2 * n)**(2/3) / rho_e # Fermi energy of Au in eV

# JDFTx parameters
V_unit_cell = 4583.13
fftbox = [56, 56, 1568]
dV = V_unit_cell / (np.prod(fftbox)) # Renormalization factor
z_dim = 154.2

# Visualization setup
x = 0.5
y = 0
z_min = 0
z_max = 0.5

x_ind = int(x * fftbox[0])
y_ind = int(y * fftbox[1])
z_ind_min = int(z_min * fftbox[2])
z_ind_max = int(z_max * fftbox[2])

# Location of no-field Kohn-Sham potential
directory = '/Users/Apple/Desktop/MNED/Data/Au_100/Ecut_test_2/Au_100_E0_Ecut50.Vscloc'
ex_corr = 'GGA-PBE' # Name of exchange-correlation

# Obtain the Kohn-Sham potential at the specified electric field
with open('/Users/Apple/Desktop/MNED/Data/Au_100/avg_elec_surf.txt','rb') as f:
    z_surf = pickle.load(f)

U_KS = np.fromfile(directory, dtype = np.float64)
U_KS = np.reshape(U_KS,fftbox)[:, :, z_ind_min:z_ind_max]
U_KS = U_KS / dV
Uz_KS = U_KS[x_ind, y_ind, :]
Uz_vacuum = np.mean(Uz_KS[-53:-3]) 
    # Take the average of the last 50 points as the vacuum potential
Uz_KS = (Uz_KS - Uz_vacuum) * Hartree_to_eV  + phi ; 
    # Conversion to eV and shift the Fermi level to E = 0

z_arr = np.linspace(z_min, z_max, num = (z_ind_max - z_ind_min)) * z_dim
dz = z_arr[1] - z_arr[0]

def Gaussian(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)
x_arr = np.linspace(-250, 249, num=500)
g_arr = Gaussian(x_arr)
Uz_KS_smooth = np.convolve(Uz_KS, g_arr, mode='same')[:-5]
# Correct for boundary effects and shift to the left by one grid point
Uz_KS_smooth = np.append(Uz_KS[:4], Uz_KS_smooth[5:])
z_arr = z_arr[:-6] # Ignore the last six points

min_ind = np.where(z_arr > z_surf)[0][0] # Minimum index where z_arr > z_surf

# Cubic spline interpolation of the KS potential
Uz_KS_cs = sp.interpolate.CubicSpline(z_arr, Uz_KS_smooth, bc_type=((1,0), (1,0)))

#% Perform image-charge fitting modification of the KS barrier
dUdz_error = np.empty(len(z_arr), dtype=float) # Error in derivatives
dUdz_percent_error = np.empty(len(z_arr), dtype=float) # Percent error in derivatives

for i in range(len(z_arr)):
    # Location of the image plane
    z_0 = z_arr[i] - Hartree_to_eV / (4 * (phi - Uz_KS_smooth[i]))
    # Analycial derivative
    dUdz = Hartree_to_eV / (4 * (z_arr[i] - dz/2 - z_0)**2)
    # Numerical derivative
    dUdz_num = (Uz_KS_smooth[i] - Uz_KS_smooth[i - 1]) / dz  
    dUdz_error[i] = dUdz - dUdz_num
    dUdz_percent_error[i] = dUdz_error[i] / dUdz_num * 100
    
z_upper_ind = 330 # Upper index for the junction (before the bump)
z_range = np.arange(301, 401) # Plotting range

# Build the image-charging fitting model with minimum percent error
z_ind_opt = np.argmin(abs(dUdz_percent_error[z_range[0]:z_upper_ind])) + z_range[0]
z_opt = z_arr[z_ind_opt]
z_0_opt = z_arr[z_ind_opt] - Hartree_to_eV / (4 * (phi - Uz_KS_smooth[z_ind_opt]))

#%% Define the potentials with vectorized argument
# KS potential with cubic interpolation extended to the real line
def Uz_KS(z):
    z = np.asarray(z)
    Uz = np.zeros(z.shape)
    Uz += (z < 0) * Uz_KS_cs(0)
    Uz += np.logical_and(z >= 0, z <= z_arr[-1]) * Uz_KS_cs(z)
    Uz += (z > z_arr[-1]) * phi
    return Uz
    
# KS potential with image charge fitting
def Uz_KS_fit(z):
    z = np.asarray(z)
    Uz = np.zeros(z.shape)
    Uz += (z <= z_opt) * Uz_KS_cs(z)
    Uz += (z > z_opt) * (phi - Hartree_to_eV/(4 * (z - z_0_opt)))
    return Uz

# ET potential with no applied E-field
def Uz_ET(z):
    z = np.asarray(z)
    Uz = np.zeros(z.shape)
    Uz += (z <= z_surf) * (-8.75)
    Uz += (z > z_surf) * phi
    return Uz

# Schottky-Nordheim potential with no applied E-field
# Note: When z = z_surf, there is a slight issue of dividing by zero. This does
#       not affect the output, though.
def Uz_SN(z):
    z = np.asarray(z)
    Uz = np.zeros(z.shape)
    Uz += (z <= z_surf ) * (-8.75)
    Uz += (z > z_surf) * np.maximum(phi - Hartree_to_eV / (4 * (z - z_surf)), -8.75)
    return Uz

# Obtain the wavefunction and its derivative assuming it has the form of Airy 
#   functions. The derivative has unit 1/Bohr
def psi_and_dev(F, E, z):
    # Inputs:
        # F - applied electric field (unit: V/m)
        # E - energy of source electrons (unit: eV)
        # z - spatial position (unit: Bohr)
    arg = -(F * (z - z_surf) * Bohr_to_m - phi + E) * rho_e * (2 * m_e)**(1/3) \
        / (h_bar * rho_e * F)**(2/3) # Argument of Airy functions
    ai, aip, bi, bip = sp.special.airy(arg)
    psi = bi + ai*1j # Wavefunction
    # Derivative of wavefunction
    psi_dev = -(2 * m_e * rho_e * F / h_bar**2)**(1/3) * Bohr_to_m * (bip + aip*1j)
    return [psi, psi_dev]

#%% Plot the four longitudinal potentials at zero field
# fig, ax = plt.subplots()
# ax.plot(z_arr, Uz_KS(z_arr), label='KS', alpha=0.7)
# ax.plot(z_arr, Uz_KS_fit(z_arr), label='IC', alpha=0.7)
# ax.plot(z_arr, Uz_ET(z_arr), label='ET', alpha=0.7)
# ax.plot(z_arr, Uz_SN(z_arr), label='SN', alpha=0.7)
# #ax.vlines(z_surf, ymin=min(Uz_KS(z_arr)), ymax=phi, colors='red', linestyles='dashed')
# ax.set_xlabel('z (Bohr radius)')
# ax.set_ylabel('Potential (eV)')
# ax.legend();

#%% Calculate the transmission coefficient using the transfer matrix method
def trans_coef_TM(z_grid, F, E, Uz):
    # Inputs:
        # z_grid - an array in z on which to evaluate the potential (unit: Bohr)
        # F - applied electric field (unit: V/m)
        # E - electron normal energy (unit: eV)
        # Uz - values of the no-field longitidunal potential on z_grid (unit: eV)
    # Outputs:
        # T - transmission coefficient
    
    if len(z_grid) != len(Uz):
        raise ValueError('z_grid and Uz should have equal length!')
        
    # Find the potential at the applied electric field
    delta_U = np.maximum(np.zeros(len(z_grid)), F * Bohr_to_m * (z_grid - z_surf))
    Uz_F = Uz - delta_U

    if E <= Uz_F[0]:
        raise ValueError('The energy of source electrons is too small!')
    
    X_arr = np.empty((2,len(z_grid)), dtype = np.complex_) # Array of wfn and its dev at each point
    X_arr[:, -1] = psi_and_dev(F, E, z_grid[-1])
    X_arr[1, -1] = X_arr[1, -1] / Bohr_to_m # Convert the derivative to metric unit

    # Back propagation
    for l in np.arange(len(z_grid)-1, 0, step = -1):
        mat = np.zeros((2,2))
        delta_z = (z_grid[l] - z_grid[l-1]) * Bohr_to_m # step in z (unit: m)
        if Uz_F[l] < E:
            k_l = np.sqrt(2 * m_e * (E - Uz_F[l]) * rho_e / h_bar**2)            
            mat = np.array([[np.cos(k_l * delta_z), -np.sin(k_l * delta_z) / k_l],
                            [k_l * np.sin(k_l * delta_z), np.cos(k_l * delta_z)]])
        elif Uz_F[l] > E:
            k_l = np.sqrt(2 * m_e * (Uz_F[l] - E) * rho_e / h_bar**2)
            mat = np.array([[np.cosh(k_l * delta_z), -np.sinh(k_l * delta_z) / k_l],
                            [-k_l * np.sinh(k_l * delta_z), np.cosh(k_l * delta_z)]])
        else:
            mat = np.array([[1, -delta_z],
                            [0, 1]])
        X_arr[:, l-1] = np.matmul(mat, X_arr[:,l]) # Wfn and dev at a prev pt
        
    # Calculate the transmission coefficient
    k_I = np.sqrt(2 * m_e * (E - Uz_F[0]) * rho_e) / h_bar # wavenumber in region I
    A_I = (X_arr[0,0] + X_arr[1,0] / (1j * k_I)) / 2 # Amplitude of forward-propagating wavefunction in region I
    J_inc = h_bar * k_I * np.abs(A_I)**2 / m_e # Incident probability current
    J_trans = (2 * h_bar * rho_e * F / m_e**2)**(1/3) / np.pi # Transmitted probability current
    T = J_trans / J_inc # Transmission coefficient
    return T

#%% Find the z_grid with increased resolution for the transfer matrix method
def get_z_grid(z_start, z_stop, mag):
    # Inputs:
        # z_start - approximate right boundary
        # z_stop - approximate left boundary
        # mag - magnification factor for the z_arr resolution
    # Output:
        # z_grid - an array in z with the specified boundaries and resolution
    z_start_ind = np.argmin(np.abs(z_arr - z_start))
    z_stop_ind = np.argmin(np.abs(z_arr - z_stop))
    
    # Increase the resolution of z_grid for the transfer-matrix method
    z_grid = np.linspace(z_stop, z_start, num = (z_start_ind - z_stop_ind) * mag + 1)
    # Index of closest point to the electrical surface
    z_surf_ind = np.argmin(abs(z_grid - z_surf))
    # Shift z_grid such as one point coincides with the electrical surface
    # Note: This step increases the accuracy of the transfer matrix method
    z_grid = z_grid - (z_grid[z_surf_ind] - z_surf)
    return z_grid

#%% Visualize the transmission coefficient versus different starting positions for KS
# z_N_ind_arr = np.arange(710, 779)
# z_1_ind = 290
# z_step = 1
# F = 1e10
# E = 0
# T_arr = np.zeros(z_N_ind_arr.size, dtype=float)
# for i in range(len(z_N_ind_arr)):
#     T_arr[i] = trans_coef_TM(z_N_ind_arr[i], z_1_ind, z_step, F, E, Uz = Uz_KS_smooth)
    
# fig1, ax1 = plt.subplots(figsize=(7.5,4.8))
# ax1.plot(z_arr[z_N_ind_arr], T_arr)
# ax1.set_xlabel('Starting position (Bohr)')
# ax1.set_ylabel('Transmission coefficient')
# ax1.set_title('Transmission coefficient versus the starting position for back-propagation\nF = %d GV/m' % (F/1e9))
# plt.subplots_adjust(left=0.16, top=0.85)

# #%% Visualize the transmission coefficient versus different ending positions for KS
# z_N_ind = -1
# z_1_ind_arr = np.arange(0, 309)
# z_step = 1
# F = 1e10
# E = 0
# T_arr = np.zeros(z_1_ind_arr.size, dtype=float)
# for i in range(len(z_1_ind_arr)):
#     T_arr[i] = trans_coef_TM(z_N_ind, z_1_ind_arr[i], z_step, F, E, Uz = Uz_KS_smooth)
    
# fig1, ax1 = plt.subplots(figsize=(7.5,4.8))
# ax1.plot(z_arr[z_1_ind_arr], T_arr)
# ax1.set_xlabel('Ending position (Bohr)')
# ax1.set_ylabel('Transmission coefficient')
# ax1.set_title('Transmission coefficient versus the ending position for back-propagation\nF = %d GV/m' % (F/1e9))
# plt.subplots_adjust(left=0.16, top=0.85)

# #%% Visualize the effects of increasing the step size for KS
# z_N_ind = -1
# z_1_ind = 290
# F_arr = np.arange(5e8, 8e10, 5e8)
# E = 0
# T_arr_dz1 = np.zeros(len(F_arr), dtype = float)
# T_arr_dz2 = np.zeros(len(F_arr), dtype = float)
# for i in range(len(F_arr)):
#     T_arr_dz1[i] = trans_coef(z_N_ind, z_1_ind, 1, F_arr[i], E, Uz = Uz_KS_smooth)
#     T_arr_dz2[i] = trans_coef(z_N_ind, z_1_ind, 2, F_arr[i], E, Uz = Uz_KS_smooth)

# fig1, ax1 = plt.subplots()
# ax1.plot(F_arr / 1e9, T_arr_dz1, label='step = 1dz')
# ax1.plot(F_arr / 1e9, T_arr_dz2, label='step = 2dz')
# ax1.set_xlabel('Applied electric field (GV/m)')
# ax1.set_ylabel('Transmission coefficient')
# ax1.legend()
# ax1.set_title('Transmission coefficient versus the applied electric field')

# fig2, ax2 = plt.subplots()
# ax2.plot(F_arr / 1e9, T_arr_dz1 / T_arr_dz2, label='step = 1dz')
# ax2.set_xlabel('Applied electric field (GV/m)')
# ax2.set_ylabel('Ratio')
# ax2.set_title('Ratio of transmission coefficients with original to doubling the step size')

# #%% Visualize the transmission coefficient at different propagating energies for KS
# z_N_ind = -1
# z_1_ind_arr = np.arange(0, 300)
# z_step = 1
# F_arr = 1e10
# E_arr = np.arange(1, -1, step = -0.2)
# T_arr = np.zeros((len(z_1_ind_arr), len(E_arr)), dtype = float)
# for i in range(len(z_1_ind_arr)):
#     for j in range(len(E_arr)):
#         T_arr[i,j] = trans_coef_TM(z_N_ind, z_1_ind_arr[i], z_step, F_arr, E_arr[j], Uz = Uz_KS_smooth)

# fig1, ax1 = plt.subplots()
# for j in range(len(E_arr)):
#     ax1.plot(z_1_ind_arr, T_arr[:,j], label='E = %.1f eV' % (E_arr[j]))
# ax1.set_xlabel('Ending position (Bohr)')
# ax1.set_ylabel('Transmission coefficient')
# ax1.legend()
# ax1.set_title('Transmission coefficient versus ending position for back-propagation\nF = %d GV/m' % (F/1e9))
# plt.yscale('log')

#%% Visualize the transfer matrix transmission coefficient versus applied 
#   electric fields for 4 potentials
# start = 76.5 # Approximate right boundary
# stop = 27.5 # Approximate left boundary
# mag = 10 # Magnification factor for the z_arr resolution
# z_grid = get_z_grid(start, stop, mag)
# F_arr = np.arange(5e8, 8e10, 5e8)
# E = 0
# Uz_arr = [Uz_KS_cs(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]
# label_arr = ['KS', 'IC', 'ET', 'SN']
# T_arr_TM = np.zeros((len(Uz_arr), len(F_arr)))

# for i in range(len(Uz_arr)):
#     for j in range(len(F_arr)):
#         T_arr_TM[i,j] = trans_coef_TM(z_grid, F_arr[j], E, Uz_arr[i])

# plt.rcParams.update({'font.size': 10})
# fig, ax = plt.subplots()
# for i in range(len(Uz_arr)):
#     ax.plot(F_arr / 1e9, T_arr_TM[i], label=label_arr[i])
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Transmission coefficient')
# ax.legend();
# ax.set_title('Transmission coefficient using the transfer-matrix method')

#%% Visualize the transfer matrix transmission coefficient versus electron
#   energy for 4 potentials
# start = 76.5 # Approximate right boundary
# stop = 27.5 # Approximate left boundary
# mag = 10 # Magnification factor for the z_arr resolution
# z_grid = get_z_grid(start, stop, mag)
# F = 8e10
# E_arr = np.arange(-2, 2, step=0.05)
# Uz_arr = [Uz_KS_cs(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]
# label_arr = ['Kohn-Sham potential', 'Image-charge corrected potential',
#               'Exact triangular potential', 'Schottky-Nordheim potential']
# T_arr = np.zeros((len(Uz_arr), len(E_arr)))

# for i in range(len(Uz_arr)):
#     for j in range(len(E_arr)):
#         T_arr[i,j] = trans_coef_TM(z_grid, F, E_arr[j], Uz_arr[i])

# fig, ax = plt.subplots()
# for i in range(len(Uz_arr)):
#     ax.plot(E_arr, T_arr[i], label=label_arr[i])
# ax.set_xlabel('Electron energy (eV)')
# ax.set_ylabel('Transmission coefficient')
# ax.set_yscale('log')
# ax.legend()
# ax.set_title('Transmission coefficient versus electron energy, F = %dGV/m' % (F/1e9));

#%% Visualize the ratio of WKB to transfer matrix transmission coefficient 
#  versus applied electric fields for 4 potentials
start = 76.5 # Approximate right boundary
stop = 27.5 # Approximate left boundary
mag = 10 # Magnification factor for the z_arr resolution
z_grid = get_z_grid(start, stop, mag)
F_arr = np.arange(5e8, 8e10, 5e8)
E = 0
Uz_func_arr = [Uz_KS, Uz_KS_fit, Uz_ET, Uz_SN]
Uz_arr = [Uz_KS(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]
mag = 1e4
label_arr = ['KS', 'IC', 'ET', 'SN']
T_arr_WKB = np.zeros((len(Uz_func_arr), len(F_arr)))
T_arr_TM = np.zeros((len(Uz_func_arr), len(F_arr)))

for i in range(len(Uz_func_arr)):
    for j in range(len(F_arr)):
        T_arr_WKB[i,j] = trans_coef_WKB(F_arr[j], E, Uz_func_arr[i], mag=mag)
        
for i in range(len(Uz_func_arr)):
    for j in range(len(F_arr)):
        T_arr_TM[i,j] = trans_coef_TM(z_grid, F_arr[j], E, Uz_arr[i])

#%%
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots()
for i in range(len(Uz_func_arr)):
    ax.plot(F_arr / 1e9, T_arr_TM[i] / T_arr_WKB[i], label=label_arr[i], alpha=0.7)
ax.set_xlabel('Applied electric field (GV/m)')
ax.set_ylabel('Tunneling pre-factor')
# ax.set_ylim([0,1])
ax.legend();
# ax.set_title('Ratio of WKB to transfer-matrix transmission coefficients');

#%% Find the decay width by numerical differentiation using the transfer matrix method
def decayWidth_numDiff_TM(z_grid, Uz, F, delta_E_init=0.1, threshold=1e-4, N=20):
    # Inputs:
        # z_grid - an array in z on which to evaluate the potential (unit: Bohr)
        # Uz - array of longitudinal potential on the z-grid (unit: eV)
        # F - applied electric field (unit: V/m)
        # delta_E - initial change in the electron energy (unit: eV)
        # threshold - convergence threshold, converged if the percentage error of
        #             two consecutive iterations differ by less than this value 
        #             (unit: eV^-1)
        # N - maximum number of iterations
    # Outputs:
        # d_F - decay width at the Fermi level (unit: eV)
        # delta_E - final change in the electron energy (unit: eV)
        # num_iter - number of iterations to converge; if not converged, num_iter = N
    
    delta_E = delta_E_init
    inv_d_F_prev = -1e5 # Arbitrarily small
    converged = False
    num_iter = 1 # Number of iterations

    while not converged and num_iter <= N:
        # Compute the transmission coefficient with lowered electron energy
        T_lower = trans_coef_TM(z_grid, F, -delta_E, Uz)
        
        # Compute the transmission coefficient with raised electron energy
        T_raise = trans_coef_TM(z_grid, F, delta_E, Uz)
        
        inv_d_F = (np.log(T_raise) - np.log(T_lower)) / (2 * delta_E)
        if abs((inv_d_F_prev - inv_d_F) / inv_d_F_prev) < threshold:
            converged = True
        else:
            inv_d_F_prev = inv_d_F
            delta_E = delta_E / 2
            num_iter = num_iter + 1
   
    if converged == False:
        raise Exception('Numerical differentiation does not converge at F = %d GV/m' % (F))
        
    d_F = 1 / inv_d_F
    
    return d_F, delta_E, num_iter

#%% Visualize the transfer-matrix decay width with respect to electric field
# start = 76.5 # Approximate right boundary
# stop = 27.5 # Approximate left boundary
# mag = 10 # Magnification factor for the z_arr resolution

# z_grid = get_z_grid(start, stop, mag)

# Uz_arr = [Uz_KS_cs(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]
# label_arr = ['KS', 'IC', 'ET', 'SN']
# F_arr = np.arange(5e8, 8e10, 5e8)
# d_F_arr_TM = np.zeros((len(Uz_arr),len(F_arr)))

# for i in range(len(Uz_arr)):
#     for j in range(len(F_arr)):
#         d_F_arr_TM[i,j] = decayWidth_numDiff_TM(z_grid, Uz_arr[i], F_arr[j])[0]
        
# plt.rcParams.update({'font.size': 12})
# fig, ax = plt.subplots()
# for i in range(len(Uz_arr)):
#     ax.plot(F_arr / 1e9, d_F_arr_TM[i], label=label_arr[i], alpha=0.7)
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Decay width (eV)')
# #ax.set_yscale('log')
# ax.set_ylim([0,6])
# ax.legend()
# ax.set_title('Transfer-matrix decay widths');

#%% Visualize the ratio of WKB to transfer-matrix decay widths
# F_arr = np.arange(5e8, 8e10, 5e8)
# F_arr_SN = np.arange(5e8, 2e10, 5e8) 
# Uz_func_arr = [Uz_KS, Uz_KS_fit, Uz_ET]
# mag = 1e4
# label_arr = ['KS', 'IC', 'ET']
# d_F_arr_WKB = np.zeros((len(Uz_func_arr),len(F_arr)))
# d_F_arr_SN_WKB = np.zeros(len(F_arr_SN))

# for i in range(len(Uz_func_arr)):
#     for j in range(len(F_arr)):
#         d_F_arr_WKB[i,j] = decayWidth_numDiff_WKB(F_arr[j], Uz_func_arr[i], mag=mag)[0]
# for j in range(len(F_arr_SN)):
#     d_F_arr_SN_WKB[j] = decayWidth_numDiff_WKB(F_arr_SN[j], Uz_SN, mag=mag)[0]

# plt.rcParams.update({'font.size': 12})
# fig, ax = plt.subplots()
# for i in range(len(Uz_func_arr)):
#     ax.plot(F_arr / 1e9, d_F_arr_WKB[i] / d_F_arr_TM[i], label=label_arr[i])
# ax.plot(F_arr_SN / 1e9, d_F_arr_SN_WKB / d_F_arr_TM[3,:len(F_arr_SN)], label='SN')
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Ratio')
# #ax.set_yscale('log')
# ax.legend()
# ax.set_title('Ratio of WKB to transfer-matrix decay widths');

#%% Find the emission currents from a potential using the transfer-matrix method
def current_TM(z_grid, F, Uz):
    # Inputs:
        # z_grid - an array in z on which to evaluate the potential (unit: Bohr)
        # F - applied electric field (unit: V/m)
        # Uz - values of the no-field longitidunal potential on z_grid (unit: eV)
    # Output:
        # J - emission current at the applied electric field (unit: A/m^2)
    T = trans_coef_TM(z_grid, F, 0, Uz) # Transmission coefficient at the Fermi level
    d_F = decayWidth_numDiff_TM(z_grid, Uz, F)[0] * rho_e # decay width in Joule
    
    # Sommerfeld's electron supply constant in SI units
    z_s = rho_e * m_e / (2 * np.pi**2 * h_bar**3) 
    
    J = z_s * T * d_F**2
    
    return J

#%% Find the emission current density from a potential using the transfer-matrix
#   method. Note that the integration is performed numerically over all electron
#   states.
def current_TM_num(z_grid, F, Uz):
    # Inputs:
        # z_grid - an array in z on which to evaluate the potential (unit: Bohr)
        # F - applied electric field (unit: V/m)
        # Uz - values of the no-field longitidunal potential on z_grid (unit: eV)
    # Output:
        # J - emission current at the applied electric field (unit: A/m^2)
    f = lambda epsilon_n: trans_coef_TM(z_grid, F, epsilon_n, Uz) * (-epsilon_n)
    # Sommerfeld's electron supply constant in SI units
    z_s = rho_e * m_e / (2 * np.pi**2 * h_bar**3)
    J = z_s * sp.integrate.quad(f, -E_F, 0)[0] * rho_e**2
    return J

#%% Generate the Fowler-Nordheim plot using the transfer-matrix method using
#   the linear approximation (decay width)
# start = 76.5 # Approximate right boundary
# stop = 27.5 # Approximate left boundary
# mag = 10 # Magnification factor for the z_arr resolution

# z_grid = get_z_grid(start, stop, mag)

# F_arr = np.arange(1e9, 8.05e10, step=1e9)
# inv_F_arr = 1 / F_arr
# Uz_arr = [Uz_KS_cs(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]

# J_arr_TM_linear = np.zeros((len(Uz_arr), len(F_arr)))
# for i in range(len(Uz_arr)):
#     for j in range(len(F_arr)):
#         J_arr_TM_linear[i,j] = current_TM(z_grid, F_arr[j], Uz_arr[i])

# label_arr = ['KS', 'IC', 'ET', 'SN']
# fig, ax = plt.subplots()
# for i in range(len(Uz_arr)):
#     ax.plot(inv_F_arr, np.log(J_arr_TM_linear[i] / F_arr**2), label=label_arr[i], alpha=0.7)
# ax.set_xlabel(r'$1/F$ (m/V)')
# ax.set_ylabel(r'ln($J/F^2$)')
# ax.legend()
# ax.set_title('Transfer-matrix Fowler-Nordheim plot')
# plt.subplots_adjust(left=0.14);

#%% Generate the Fowler-Nordheim plot using the transfer-matrix method and 
#   numerical integration
# start = 76.5 # Approximate right boundary
# stop = 27.5 # Approximate left boundary
# mag = 10 # Magnification factor for the z_arr resolution

# z_grid = get_z_grid(start, stop, mag)

# F_arr = np.arange(1e9, 8.05e10, step=1e9)
# inv_F_arr = 1 / F_arr
# Uz_arr = [Uz_KS_cs(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]

# J_arr_TM = np.zeros((len(Uz_arr), len(F_arr)))
# for i in range(len(Uz_arr)):
#     for j in range(len(F_arr)):
#         J_arr_TM[i,j] = current_TM_num(z_grid, F_arr[j], Uz_arr[i])

# label_arr = ['KS', 'IC', 'ET', 'SN']
# fig, ax = plt.subplots()
# for i in range(len(Uz_arr)):
#     ax.plot(inv_F_arr, np.log(J_arr_TM[i] / F_arr**2), label=label_arr[i], alpha=0.7)
# ax.set_xlabel(r'$1/F$ (m/V)')
# ax.set_ylabel(r'ln($J/F^2$)')
# ax.legend()
# ax.set_title('Transfer-matrix Fowler-Nordheim plot')
# plt.subplots_adjust(left=0.14);

#%% Plot the correction factor to using the decay width (ratio of numerical 
#   integration to linear approximation)
# fig, ax = plt.subplots()
# for i in range(len(Uz_arr)):
#     ax.plot(F_arr / 1e9, J_arr_TM[i] / J_arr_TM_linear[i], label=label_arr[i], alpha=0.7)
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Correction factor')
# #ax.set_yscale('log')
# ax.legend();

#%% Plot the ratio of WKB to transfer-matrix emission currents
# F_arr = np.arange(1e9, 8e10, 5e8)
# F_arr_SN = np.arange(1e9, 2e10, 5e8) 
# E = 0
# Uz_func_arr = [Uz_KS, Uz_KS_fit, Uz_ET]
# mag = 1e4
# label_arr = ['KS', 'IC', 'ET']
# J_arr_WKB = np.zeros((len(Uz_func_arr),len(F_arr)))
# J_arr_SN_WKB = np.zeros(len(F_arr_SN))

# for i in range(len(Uz_func_arr)):
#     for j in range(len(F_arr)):
#         J_arr_WKB[i,j] = current_WKB(F_arr[j], E, Uz_func_arr[i], mag=mag)
# for j in range(len(F_arr_SN)):
#     J_arr_SN_WKB[j] = current_WKB(F_arr_SN[j], E, Uz_SN, mag=mag)

# fig, ax = plt.subplots()
# for i in range(len(Uz_func_arr)):
#     ax.plot(F_arr / 1e9, J_arr_WKB[i] / J_arr_TM[i], label=label_arr[i], alpha=0.7)
# ax.plot(F_arr_SN / 1e9, J_arr_SN_WKB / J_arr_TM[3,:len(F_arr_SN)], label='SN', alpha=0.7)
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Ratio')
# #ax.set_yscale('log')
# ax.set_ylim([0,2])
# ax.legend()
# ax.set_title('Ratio of WKB to transfer-matrix emission currents');
