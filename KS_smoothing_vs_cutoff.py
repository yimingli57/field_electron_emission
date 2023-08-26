#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:37:50 2022

@author: apple
"""

# Smooths out the oscillations in the Kohn-Sham potential with different charge density cutoffs

#%% Setup

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Ecut_arr = ['20', '25', '30', '35', '40', '45', '50', '55']
# fftbox_arr = [[36, 36, 1000],
#               [40, 40, 1120],
#               [48, 48, 1250],
#               [48, 48, 1344],
#               [54, 54, 1400],
#               [56, 56, 1500],
#               [56, 56, 1568],
#               [60, 60, 1680]]
Ecut_arr = ['45', '50', '55', '60', '65', '70']
fftbox_arr = [[56, 56, 1500],
              [56, 56, 1568],
              [60, 60, 1680],
              [64, 64, 1728],
              [64, 64, 1792],
              [70, 70, 1890]]
#
V_unit_cell = 4583.13

x = 0.5
y = 0
z_dim = 154.2
z_min = 0
z_max = 0.5

# Location of output files and parameters used in DFT calculation
directory = '/Users/Apple/Desktop/MNED/Data/Au_100_potential/Ecut_test_E100/'
file_name = 'Au_100_'
ex_corr = 'GGA-PBE' # Name of exchange-correlation
e_field = '100 MV/m' # Strenght of applied electric field

enableSmoothing = True # Whether smoothing with Weierstrass transform is enabled

#%% Data processing and plotting
potentials_z_pre = 0;

for i in range(len(Ecut_arr)):
    x_ind = int(x * fftbox_arr[i][0])
    y_ind = int(y * fftbox_arr[i][1])
    z_ind_min = int(z_min * fftbox_arr[i][2])
    z_ind_max = int(z_max * fftbox_arr[i][2])
    
    # Import the Kohn-Sham potential
    potentials = np.fromfile(directory + file_name + Ecut_arr[i] + '.Vscloc', dtype = np.float64)
    potentials = potentials.reshape(fftbox_arr[i])[:, :, z_ind_min:z_ind_max]
    potentials_z = potentials[x_ind, y_ind, :] # Plot the potential along a normal line
    # potentials_z = np.mean(potentials, axis=(0,1)) # Plot the potential averaged over the xy-plane
    dV = V_unit_cell / (np.prod(fftbox_arr[i])) # Renormalization factor
    potentials_z = potentials_z / dV
    
    z_arr = np.linspace(z_min, z_max, num = (z_ind_max - z_ind_min)) * z_dim
    
    if enableSmoothing:
        def Gaussian(x):
            return np.exp(-x**2/2)/np.sqrt(2*np.pi)
        x_arr = np.linspace(-250, 249, num=500)
        g_arr = Gaussian(x_arr)
        potentials_z_smooth = np.convolve(potentials_z, g_arr, mode='same')
        plt.plot(z_arr, potentials_z_smooth, alpha = 0.7, label = 'Ecut={}'.format(Ecut_arr[i]))
        if i > 0: # Error analysis
            potentials_z_diff = potentials_z_pre(z_arr) - potentials_z_smooth
            std = np.sqrt(np.sum((potentials_z_diff)**2) / (z_ind_max - z_ind_min))
            L_inf = np.max(abs(potentials_z_diff))
            print('Ecut = {}: std = {}, L_inf = {}\n'.format(Ecut_arr[i], std, L_inf))
        potentials_z_pre = CubicSpline(z_arr, potentials_z_smooth)
            
    else:    
        plt.plot(z_arr, potentials_z, alpha = 0.7, label = 'Ecut={}'.format(Ecut_arr[i]))
        if i > 0: # Error analysis
            potentials_z_diff = potentials_z_pre(z_arr) - potentials_z
            std = np.sqrt(np.sum((potentials_z_diff)**2) / (z_ind_max - z_ind_min))
            L_inf = np.max(abs(potentials_z_diff))
            print('Ecut = {}: std = {}, L_inf = {}\n'.format(Ecut_arr[i], std, L_inf))
        potentials_z_pre = CubicSpline(z_arr, potentials_z)

plt.legend()
plt.xlabel('z (a.u.)')
plt.ylabel('Potential (Hartree)')
plt.title('Longitudinal Smoothed Potential (E = ' + e_field + '), kpoint = 20')