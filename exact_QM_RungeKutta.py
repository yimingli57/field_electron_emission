#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 17:32:27 2023

@author: apple
"""

import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Desktop.MNED.Code.exact_QM_transferMatrix import trans_coef_TM
from Desktop.MNED.Code.exact_QM_transferMatrix import get_z_grid

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

#%% Obtain the Kohn-Sham potential at the specified electric field
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

#%% Plot the no-field and with-field KS potentials
# F = 1e10
# delta_U = np.maximum(0, 1e10 * Bohr_to_m * (z_arr - z_surf))
# fig, ax = plt.subplots()
# ax.plot(z_arr, Uz_KS_smooth, label = 'F = 0')
# ax.plot(z_arr, Uz_KS_smooth - delta_U, label = 'F = 10 GV/m')
# ax.scatter(z_surf, Uz_KS_cs(z_surf), label = 'Electrical surface', color = 'purple')
# ax.set_xlabel('z (Bohr radius)')
# ax.set_ylabel('Potential (eV)')
# ax.legend()
# ax.set_title('No-field vs with-field Kohn-Sham potentials')

#%% Perform image-charge fitting modification of the KS barrier
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

# KS potential with image charge fitting
def Uz_KS_fit(z):
    z = np.asarray(z)
    Uz = np.zeros(z.shape)
    Uz += (z <= z_opt) * Uz_KS_cs(z)
    Uz += (z > z_opt) * (phi - Hartree_to_eV/(4 * (z - z_0_opt)))
    return Uz

#%% Plot the image-charge corrected potential
# fig, ax = plt.subplots()
# ax.plot(z_arr, Uz_KS_smooth, label='Kohn-Sham potential')
# ax.plot(z_arr, Uz_KS_fit(z_arr), label='Image-charge corrected potential')
# ax.scatter(z_opt, Uz_KS_cs(z_opt), label='Optimal fitting point', color='purple')
# ax.set_xlabel('z (Bohr radius)')
# ax.set_ylabel('Potential (eV)')
# ax.legend()
# ax.set_title('Kohn-Sham potential and image-charge corrected potential')

#%% Plot the four potentials with no electric field
fig, ax = plt.subplots()
ax.plot(z_arr, Uz_KS_cs(z_arr), label = 'KS', alpha = 0.7)
ax.plot(z_arr, Uz_KS_fit(z_arr), label = 'IC', alpha = 0.7)
ax.plot(z_arr, Uz_ET(z_arr), label = 'ET', alpha = 0.7)
ax.plot(z_arr, Uz_SN(z_arr), label = 'SN', alpha = 0.7)
ax.legend()
ax.set_xlabel('z (Bohr radius)')
ax.set_ylabel('Potential (eV)')
ax.set_title('Four types of potentials, F = 0')
plt.rcParams.update({'font.size': 14});

#%% Plot the four potentials with an electric field
F = 1e10
fig, ax = plt.subplots()
delta_U = np.maximum(0, F * Bohr_to_m * (z_arr - z_surf))
ax.plot(z_arr, Uz_KS_cs(z_arr) - delta_U, label = 'KS', alpha = 0.7)
ax.plot(z_arr, Uz_KS_fit(z_arr) - delta_U, label = 'IC', alpha = 0.7)
ax.plot(z_arr, Uz_ET(z_arr) - delta_U, label = 'ET', alpha = 0.7)
ax.plot(z_arr, Uz_SN(z_arr) - delta_U, label = 'SN', alpha = 0.7)
ax.legend()
ax.set_xlabel('z (Bohr radius)')
ax.set_ylabel('Potential (eV)')
ax.set_title('Four types of potentials, F = %dGV/m' % (F / 1e9))
plt.rcParams.update({'font.size': 14});

#%% Obtain the wavefunction and its derivative assuming it has the form of Airy 
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
    
#%% Numerically solve the TISE with initial conditions
def solve_TISE(start, stop, F, E, Uz_func, method):
    # Inputs:
        # start - right boundary of the interval (unit: Bohr)
        # stop - left boundary of the interval (unit: Bohr)
        # F - applied electric field (unit: V/m)
        # E - energy of source electrons (unit: eV)
        # Uz_func - function of the longitidunal potential for a continuous z (unit: eV)
        # method- numerical method for solving the IVP
    # Outputs:
        # sol - solution to the TISE with initial conditions specified as Airy functions
        # Uz_F - longitudinal potential with applied E-field (unit: eV) as a 
        #        continuous function of z in Bohr

    # Find the potential at the applied electric field
    delta_U = lambda z: np.maximum(0, F * Bohr_to_m * (z - z_surf))
    Uz_F = lambda z: Uz_func(z) - delta_U(z)
    
    if E <= Uz_F(stop):
        raise ValueError('The energy of source electrons is too small')

    def f(z, y): return [y[1], -2*m_e/(h_bar**2) * Bohr_to_m**2 * (E - Uz_F(z)) * rho_e * y[0]]
    # Maximum step size is set to one grid-point separation
    sol = sp.integrate.solve_ivp(f, [start, stop], psi_and_dev(F, E, start),
                                 method=method, max_step=dz)
    return sol, Uz_F

# Calculate the transmission coefficient
def trans_coef(start, stop, F, E, Uz_func, method):
    # Inputs:
        # start - right boundary of the interval (unit: Bohr)
        # stop - left boundary of the interval (unit: Bohr)
        # F - applied electric field (unit: V/m)
        # E - energy of source electrons (unit: eV)
        # Uz_func - function of the longitidunal potential for a continuous z (unit: eV)
        # method- numerical method for solving the IVP
    # Outputs:
        # T - transmission coefficient    
        
    sol, Uz_F = solve_TISE(start, stop, F, E, Uz_func, method)
    k_I = np.sqrt(2 * m_e * (E - Uz_F(stop)) * rho_e) / h_bar # wavenumber in region I
    A_I = (sol.y[0,-1] + sol.y[1,-1] / (1j * k_I * Bohr_to_m)) / 2 # Amplitude of forward-propagating wavefunction in region I
    J_inc = h_bar * k_I * np.abs(A_I)**2 / m_e # Incident probability current
    J_trans = (2 * h_bar * rho_e * F / m_e**2)**(1/3) / np.pi # Transmitted probability current
    T = J_trans / J_inc # Transmission coefficient
    return T

#%% Visualize the transmission coefficient versus starting position for KS
# start_arr = np.linspace(70, 76.5)
# stop = 27.5
# F = 1e10
# E = 0
# T_arr = np.zeros(start_arr.size)
# for i in range(len(start_arr)):
#     T_arr[i] = trans_coef(start_arr[i], stop, F, E, Uz_KS_cs)
# fig, ax = plt.subplots(figsize=(7.5,4.8))
# ax.plot(start_arr, T_arr)
# ax.set_xlabel('Right boundary (Bohr)')
# ax.set_ylabel('Transmission coefficient')
# ax.set_title('Transmission coefficient versus the right boundary\nF = %d GV/m, Runge-Kutta' % (F/1e9))
# plt.subplots_adjust(left=0.16, top=0.85)

#%% Visualize the transmission coefficient versus stopping position for KS, at
#   different propagating energies
# start = 76.5
# stop_arr = np.linspace(0, 29.5, num=100)
# F = 1e10
# E_arr = np.arange(1, -1, step = -0.2)
# T_arr = np.zeros((len(stop_arr), len(E_arr)))
# for i in range(len(stop_arr)):
#     for j in range(len(E_arr)):
#         T_arr[i,j] = trans_coef(start, stop_arr[i], F, E_arr[j], Uz_KS_cs)
# fig, ax = plt.subplots()
# for j in range(len(E_arr)):
#     ax.plot(stop_arr, T_arr[:,j], label='E = %.1f eV' % (E_arr[j]))
# ax.set_xlabel('Left boundary (Bohr)')
# ax.set_ylabel('Transmission coefficient')
# ax.set_title('Transmission coefficient versus the left boundary\nF = %d GV/m, Runge-Kutta' % (F/1e9))
# ax.legend()

#%% Visualize the Runge-Kutta transmission coefficient versus applied electric field for 4 potentials
# start = 76.5
# stop = 27.5
# F_arr = np.arange(5e8, 8e10, 5e8)
# E = 0
# method = 'DOP853'
# Uz_func_arr = [Uz_ET, Uz_SN, Uz_KS_cs, Uz_KS_fit]
# label_arr = ['ET potential', 'SN potential', 'KS potential', 'Image-charge fitted potential']
# T_arr = np.zeros((len(Uz_func_arr), len(F_arr)))

# for i in range(len(Uz_func_arr)):
#     for j in range(len(F_arr)):
#         T_arr[i,j] = trans_coef(start, stop, F_arr[j], E, Uz_func_arr[i], method)

# fig1, ax1 = plt.subplots()
# for i in range(len(Uz_func_arr)):
#     ax1.plot(F_arr / 1e9, T_arr[i,:], label=label_arr[i])
# ax1.set_xlabel('Applied electric field (GV/m)')
# ax1.set_ylabel('Transmission coefficient')
# ax1.legend()
# ax1.set_title('Transmission coefficient for four types of potentials, %s' % (method))

#%% Visualize the ratio between transfer-matrix and Runge-Kutta transmission 
#   coefficient at various applied electric fields for 4 potentials
start = 76.5 # Approximate right boundary
stop = 27.5 # Approximate left boundary
mag = 10 # Magnification factor for the z_arr resolution

z_grid = get_z_grid(start, stop, mag)

F_arr = np.arange(5e8, 8e10, 5e8)
E = 0.5
method = 'DOP853'
Uz_func_arr = [Uz_KS_cs, Uz_KS_fit, Uz_ET, Uz_SN]
Uz_grid = [Uz_KS_cs(z_grid), Uz_KS_fit(z_grid), Uz_ET(z_grid), Uz_SN(z_grid)]
label_arr = ['KS', 'IC', 'ET', 'SN']
T_arr = np.zeros((len(Uz_func_arr), len(F_arr)))
T_TM_arr = np.zeros((len(Uz_func_arr), len(F_arr)))

for i in range(len(Uz_func_arr)):
    for j in range(len(F_arr)):
        T_arr[i,j] = trans_coef(start, stop, F_arr[j], E, Uz_func_arr[i], method)
        T_TM_arr[i,j] = trans_coef_TM(z_grid, F_arr[j], E, Uz_grid[i])

#%%
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots()
for i in range(len(Uz_func_arr)):
    ax.plot(F_arr / 1e9, T_TM_arr[i] / T_arr[i], label=label_arr[i], alpha=0.7)
ax.set_xlabel('Applied electric field (GV/m)')
ax.set_ylabel('Ratio')
ax.legend()
#ax.set_title('Ratio of transfer-matrix to %s transmission coefficients' % (method));
