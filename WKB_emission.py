#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:29:48 2022

@author: Yiming Li

Modify the Kohn-Sham potential; 
Derive a general potential based on the no-field Kohn-Sham potential; 
Compare emission using the Kohn-Sham, Exact-Triangular and Schottky-Nordheim
potentials
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pickle

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

# Import the Kohn-Sham potential
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

# Perform the generalized Weierstrass transform using numpy.convolve() to 
# smooth out the high-frequency oscillations in the Kohn-Sham potential
with open('/Users/Apple/Desktop/MNED/Data/Au_100/avg_elec_surf.txt','rb') as f:
    z_surf = pickle.load(f)

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

# Perform image-charge fitting modification of the KS barrier
dz = z_arr[1] - z_arr[0]
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
Uz_KS_fit = np.append(Uz_KS_smooth[:z_ind_opt], 
                      -Hartree_to_eV/(4 * (z_arr[z_ind_opt:] - z_0_opt)) + phi)

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

#%% Calculate the transmission coefficients under the WKB approximation
def trans_coef_WKB(F, E, Uz_func, mag=1e3):
    # Inputs:
        # F - applied electric field (unit: V/m)
        # E - energy of source electrons (unit: eV)
        # Uz_func - analytical expression of the longitudinal potential without
        #           applied E-field (unit: eV)
        # mag - magnification factor for the z-grid resolution
    # Output:
        # T - transmission coefficient given by the WKB approximation

    Uz_F = lambda z: Uz_func(z) - E - (z > z_surf) * F * Bohr_to_m * (z - z_surf)
    res = np.where(Uz_F(z_arr) > 0)[0]
    
    if len(res) != 0:
        lower_lim = sp.optimize.root_scalar(Uz_F, method='bisect',
                                             bracket=[z_arr[res[0]] - dz, z_arr[res[0]]]).root
        z_U0 = z_surf + (phi - E) / (F * Bohr_to_m)
        upper_lim = sp.optimize.root_scalar(Uz_F, method='secant',
                                              x0=z_U0, x1=z_U0+1).root
        integrand = lambda z: (Uz_F(z) * rho_e)**(1/2)
        z_grid = np.arange(lower_lim+1e-8, upper_lim-1e-8, dz/mag)
        integral = sp.integrate.trapezoid(integrand(z_grid), z_grid)
        G = 2 * np.sqrt(2 * m_e) / h_bar * integral * Bohr_to_m
    else:
        G = 0
    
    T = np.exp(-G)
    
    return T

#%% Plot the WKB transmission coefficients at various applied electric fields 
#   for 4 potentials
# F_arr = np.arange(5e8, 8e10, 5e8)
# E = 0
# Uz_func_arr = [Uz_KS, Uz_KS_fit, Uz_ET, Uz_SN]
# label_arr = ['Kohn-Sham potential', 'Image-charge corrected potential',
#               'Exact triangular potential', 'Schottky-Nordheim potential']
# T_WKB_arr = np.zeros((len(Uz_func_arr),len(F_arr)))
# for i in range(len(Uz_func_arr)):
#     for j in range(len(F_arr)):
#         T_WKB_arr[i,j] = trans_coef_WKB(F_arr[j], E, Uz_func=Uz_func_arr[i], mag=1e4)

# fig, ax = plt.subplots()
# for i in range(len(Uz_func_arr)):
#     ax.plot(F_arr / 1e9, T_WKB_arr[i], label=label_arr[i])
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Transmission coefficient')
# ax.legend(loc = 'lower right')
# ax.set_title('WKB transmission coefficients')
# ax.set_ylim([-0.05, 1.05]);

#%% Plot the WKB transmission coefficients as a function of electron energy
# F = 3e10
# E_arr = np.arange(-2, 2, step=0.01)
# T_arr = np.zeros(len(E_arr))
# for i in range(len(E_arr)):
#     T_arr[i] = trans_coef_WKB(F, E_arr[i], Uz_func = Uz_SN)
# fig, ax = plt.subplots()
# ax.plot(E_arr, T_arr)
# ax.set_xlabel('Electron energy (eV)')
# ax.set_ylabel('Transmission coefficient')
# ax.set_yscale('log')
# ax.set_title('Transmission coefficient vs electron energy using WKB');

#%% Find the inverse decay width by numerical differentiation using the WKB approximation
def decayWidth_numDiff_WKB(F, Uz_func, mag=1e3, delta_E_init=0.1, threshold=1e-4, N=20):
    # Inputs:
        # F - applied electric field (unit: V/m)
        # Uz_func - analytical expression of the longitudinal potential without
        #           applied E-field (unit: eV)
        # mag - magnification factor for the z-grid resolution
        # delta_E_init - initial change in the electron energy
        # threshold - convergence threshold, converged if the percentage error of
        #             two consecutive iterations differ by less than this value 
        #             (unit: eV^-1)
        # N - maximum number of iterations
    # Outputs:
        # d_F - decay width at the Fermi level (unit: eV)
        # delta_h - final change in h
        # num_iter - number of iterations to converge; if not converged, num_iter = N
        
    delta_E = delta_E_init
    inv_d_F_prev = -1e5 # Arbitrarily small
    converged = False
    num_iter = 1 # Number of iterations

    while not converged and num_iter <= N:
        # Compute the transmission coefficient with lowered electron energy
        T_lower2 = trans_coef_WKB(F, -2*delta_E, Uz_func = Uz_func, mag=mag)
        T_lower1 = trans_coef_WKB(F, -1*delta_E, Uz_func = Uz_func, mag=mag)
        # Compute the transmission coefficient with raised electron energy
        T_raise1 = trans_coef_WKB(F,  1*delta_E, Uz_func = Uz_func, mag=mag)
        T_raise2 = trans_coef_WKB(F,  2*delta_E, Uz_func = Uz_func, mag=mag)

        inv_d_F = (np.log(T_lower2) - 8 * np.log(T_lower1) + \
                   8 * np.log(T_raise1) - np.log(T_raise2)) / (12 * delta_E)
        if abs((inv_d_F_prev - inv_d_F) / inv_d_F_prev) < threshold:
            converged = True
        else:
            inv_d_F_prev = inv_d_F
            delta_E = delta_E / 2
            num_iter = num_iter + 1
       
    if converged == False:
        raise Exception('Numerical differentiation does not converge at F = %.1f GV/m' % (F/1e9))
        
    d_F = 1 / inv_d_F
    
    return d_F, delta_E, num_iter

#%% Plot the WKB decay widths at various applied electric fields for 4 potentials
# F_arr = np.arange(5e8, 8e10, 5e8)
# F_arr_SN = np.arange(5e8, 2e10, 5e8)
# # Note: The transmission coefficient of SN peaks at 1 for F > 20GV/m
# Uz_func_arr = [Uz_KS, Uz_KS_fit, Uz_ET]
# label_arr = ['KS', 'IC', 'ET']
# d_F_WKB_arr = np.zeros((len(Uz_func_arr),len(F_arr)))
# d_F_WKB_SN = np.zeros(len(F_arr_SN))
# for i in range(len(Uz_func_arr)):
#     for j in range(len(F_arr)):
#         d_F_WKB_arr[i,j] = decayWidth_numDiff_WKB(F_arr[j], Uz_func_arr[i], mag=1e4)[0]
# for j in range(len(F_arr_SN)):
#     d_F_WKB_SN[j] = decayWidth_numDiff_WKB(F_arr_SN[j], Uz_SN, mag=1e4)[0]
# #%%
# plt.rcParams.update({'font.size': 14});
# fig, ax = plt.subplots()
# for i in range(len(Uz_func_arr)):
#     ax.plot(F_arr / 1e9, d_F_WKB_arr[i], label=label_arr[i], alpha=0.7)
# ax.plot(F_arr_SN / 1e9, d_F_WKB_SN, label='SN', alpha=0.7)
# ax.set_xlabel('Applied electric field (GV/m)')
# ax.set_ylabel('Decay width (eV)')
# ax.set_ylim([0,6])
# ax.set_title('WKB decay widths')
# ax.legend();

#%% Find the emission currents from a potential barrier using the WKB approximation
def current_WKB(F, E, Uz_func, mag=1e3):
    # Inputs:
        # z_arr - 1D array of z-positions (unit: Bohr)
        # E_field - applied electric field
        # Uz_func - analytical expression of the longitudinal potential without
        #           applied E-field (unit: eV)
        # mag - magnification factor for the z-grid resolution
    # Outputs:
        # current - emission current at the applied E-field 
    T = trans_coef_WKB(F, E, Uz_func, mag=mag)
    d_F = decayWidth_numDiff_WKB(F, Uz_func, mag=mag)[0] * rho_e # decay width in Joule
    
    # Sommerfeld's electron supply constant in SI units
    z_s = rho_e * m_e / (2 * np.pi**2 * h_bar**3) 
    
    J = z_s * T * d_F**2
    
    return J

#%%
# Emission currents from the ET barrier at an applied E-field
# def current_ET(E_field):
#     a = 1.541433e-6 # First Fowler-Nordheim constant in A*eV/V^2
#     b = 6.830890e9 # Second Fowler-Nordheim constant in eV^(-3/2)*V/m
#     # Current in A/m^2
#     current = a * E_field**2 / phi * np.exp(-b * phi**(3/2) / E_field)
#     return current

#%% Generate Fowler-Nordheim plots
# F_arr = np.arange(8e10, 5e8, step=-5e8)
# inv_F_arr = 1 / F_arr
# F_arr_SN = np.arange(2e10, 5e8, step=-5e8) 
# inv_F_arr_SN = 1 / F_arr_SN
# # Note: The transmission coefficient of SN peaks at 1 for F > 20GV/m
# E = 0
# Uz_func_arr = [Uz_KS, Uz_KS_fit, Uz_ET]
# label_arr = ['KS', 'IC', 'ET']

# J_WKB_arr = np.zeros((len(Uz_func_arr),len(F_arr)))
# J_WKB_arr_SN = np.zeros(len(F_arr_SN))

# for i in range(len(Uz_func_arr)):
#     for j in range(len(F_arr)):
#         J_WKB_arr[i,j] = current_WKB(F_arr[j], E, Uz_func_arr[i], mag=1e4)
# for j in range(len(F_arr_SN)):
#     J_WKB_arr_SN[j] = current_WKB(F_arr_SN[j], E, Uz_SN, mag=1e4)

#%%
# plt.rcParams.update({'font.size': 12});
# fig, ax = plt.subplots()
# for i in range(len(Uz_func_arr)):
#     ax.plot(inv_F_arr, np.log(J_WKB_arr[i] / F_arr**2), label=label_arr[i], alpha = 0.7)
# ax.plot(inv_F_arr_SN, np.log(J_WKB_arr_SN / F_arr_SN**2), label='SN', alpha = 0.7)
# ax.set_xlabel(r'$1/F$ (m/V)')
# ax.set_ylabel(r'ln($J/F^2$)')
# ax.legend()
# ax.set_title('WKB Fowler-Nordheim plot')
# plt.subplots_adjust(left=0.14);

#%% Find the slope of the Fowler-Nordheim plots
# Build a linear regression model and find its slope and intercept
# def linreg_model(X, y):
#     A = np.vstack([X, np.ones(len(X))]).T
#     m, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     return m, c

# m_KS, c_KS = linreg_model(inv_E_fields_arr, y_KS_arr)
# m_ET, c_ET = linreg_model(inv_E_fields_arr, y_ET_arr)
# m_SN, c_SN = linreg_model(inv_E_fields_arr, y_SN_arr)
# m_KS_fit, c_KS_fit = linreg_model(inv_E_fields_arr, y_KS_fit_arr)

# print('Slope of ET compared to slope of KS is %.4f' % (m_ET / m_KS))
# print('Slope of SN compared to slope of KS is %.4f' % (m_SN / m_KS))
# print('Slope of KS fitted compared to slope of KS is %.4f' % (m_KS_fit / m_KS))
# print('Intercept of ET is %.4f' % (c_ET))
# print('Intercept of KS is %.4f' % (c_KS))
# print('Intercept of SN is %.4f' % (c_SN))
# print('Intercept of KS fitted is %.4f' % (c_KS_fit))
# Note: Slope of ET = b*phi**(3/2)
#       Intercetp of ET = np.log(a/phi), phi in SI unit
