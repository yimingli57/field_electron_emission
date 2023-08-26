#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 23:00:23 2022

@author: apple
"""

# Plot the Kohn-Sham potentials with and without an applied electric field

import numpy as np
import matplotlib.pyplot as plt

grid = [56, 56, 1568]
x = 0.5
x_ind = int(x * grid[0])
y = 0
y_ind = int(y * grid[1])
z_dim = 154.2
z_min = 0
z_max = 0.5
z_ind_min = int(z_min * grid[2])
z_ind_max = int(z_max * grid[2])

# Define the z-location of each Au atom
ionpos = np.array([0.000,
                   0.025,
                   0.050,
                   0.075,
                   0.100,
                   0.125,
                   0.150,
                   0.175])
ionpos *= z_dim

directory = '/Users/Apple/Desktop/MNED/Data/Au_100_potential/'
# directory = '/Users/Apple/Downloads/'
file_name = 'Ecut_test_2/Au_100_E0_Ecut50'
file_name_E = 'Ecut_test_E100/Au_100_50'

# Import the Kohn-Sham potential for no field calculation
potentials_0 = np.fromfile('/Users/Apple/Desktop/MNED/Data/Au_100_potential/Ecut_test_2/Au_100_E0_Ecut50.Vscloc', dtype = np.float64)
potentials_0 = potentials_0.reshape(grid)[:, :, z_ind_min:z_ind_max]
potentials_z_0 = potentials_0[x_ind, y_ind, :]

# Import the total electrostatic potential for calculation with E = 100 MV/m
potentials_100 = np.fromfile('/Users/Apple/Desktop/MNED/Data/Au_100_potential/Ecut_test_E100/Au_100_50.Vscloc', dtype = np.float64)
potentials_100 = potentials_100.reshape(grid)[:, :, z_ind_min:z_ind_max]
potentials_z_100 = potentials_100[x_ind, y_ind, :]

# Import the total electrostatic potential for calculation with E = 500 MV/m
# potentials_500 = np.fromfile('/Users/Apple/Downloads/Au_100_E500.Vscloc', dtype = np.float64)
# potentials_500 = potentials_500.reshape(grid)[:, :, z_ind_min:z_ind_max]
# potentials_z_500 = potentials_500[x_ind, y_ind, :]

z = np.linspace(z_min, z_max, num = (z_ind_max - z_ind_min)) * z_dim

#Perform the Weierstrass transform using numpy.convolve()
def Gaussian(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)
x_arr = np.linspace(-250, 249, num=500)
g_arr = Gaussian(x_arr)

potentials_z_0_smooth = np.convolve(potentials_z_0, g_arr, mode='same')
potentials_z_100_smooth = np.convolve(potentials_z_100, g_arr, mode='same')

# Plotting
# plt.plot(z, potentials_z_0, color = 'red', label = 'E = 0')
# plt.plot(z, potentials_z_100, color = 'blue', label = 'E = 100 MV/m')
plt.plot(z, potentials_z_100_smooth - potentials_z_0_smooth, color = 'orange', label = 'delta U')
# plt.plot(z, potentials_z_500, color = 'cyan', label = 'E = 500 MV/m')
# plt.scatter(ionpos, [-0.0045, -0.0045, -0.0045, -0.0045, -0.0045, -0.0045, -0.0045, -0.0045], color = 'orange', marker = 'o', label = 'atom')
# for i in range(0,8):
#     plt.vlines(ionpos[i], -0.006,0.0003, colors = 'orange', linestyles = 'dashed')
# plt.axhline(0, color = 'black')
plt.xlabel('z (a.u.)')
plt.ylabel('Kohn-Sham Potential (Hartree)')
plt.legend()
plt.title('Kohn-Sham Potential with and without Electric Field')

