#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:50:26 2023

@author: apple
"""

# Find the repulsion distance of Au (100) using JDFTx electron densities

import numpy as np
import matplotlib.pyplot as plt

# Location of JDFTx output files
directory = '/Users/Apple/Desktop/MNED/Data/Au_100/'
file_names = ['Ecut_test_2/Au_100_E0_Ecut50.n','E_test/Au_100_300MV_fix.n']

# JDFTx parameters
fftbox = [56, 56, 1568]
e_fields = [0, 1e8] # Applied electric field of each file
z_dim = 154.2

# Visualization setup
# x = 0.5
# y = 0
z_min = 0
z_max = 0.5

# x_ind = int(x * fftbox[0])
# y_ind = int(y * fftbox[1])
z_ind_min = int(z_min * fftbox[2])
z_ind_max = int(z_max * fftbox[2])

# Import the charge densities
rho_arr = np.empty((np.prod(fftbox),len(file_names)), dtype=float) # Electron densities
for i in range(len(file_names)):
    rho_arr[:,i] = np.fromfile(directory + file_names[i], dtype = np.float64)
rho_arr = np.reshape(rho_arr,(fftbox + [len(file_names)]))[:,:,z_ind_min:z_ind_max,:]
rho_induced = rho_arr[:,:,:,1] - rho_arr[:,:,:,0] # Induced electron density

# Ionic positions of the outermost three atoms
ion_pos_z = np.array([[0.178549989836359,0.152592585495530,0.126380934604219],
                      [0.178650015778382,0.152643605644275,0.126393443379473]]) * z_dim

#%% Plot the longitudinal induced charge density
z_arr = np.linspace(z_min, z_max, num = (z_ind_max - z_ind_min)) * z_dim
rho_induced_z = np.mean(rho_induced,axis=(0,1)) # Average over the xy-plane

fig1,ax1 = plt.subplots()
ax1.plot(z_arr,-rho_induced_z,label='Induced charge density')
ax1.scatter(ion_pos_z[0,:],np.zeros(3),color='orange',label='E = 0',alpha=0.5)
# ax1.scatter(ion_pos_z[1,:],np.zeros(3),color='green',label='E = %d MV/m' % 
#             (e_fields[1]/1e6),alpha=0.5)
ax1.legend()
ax1.set_xlabel('z (Bohr)')
ax1.set_ylabel('Charge density (e)')
ax1.set_title('Averaged induced charge density at E = %d MV/m' % (e_fields[1]/1e6))

#%% Find the centroid of induced charge
# Bounds of integration are determined by the indices of above-threshold induced rho, 
# also checking the plot above
z_lower_ind = 0 # Inclusive
z_upper_ind = len(z_arr) # Exclusive

dz = z_arr[1] - z_arr[0] # Integration step
rho_induced_z_chopped = rho_induced_z[z_lower_ind:z_upper_ind]
z_chopped = z_arr[z_lower_ind:z_upper_ind]
centroid_z = np.sum(rho_induced_z_chopped*z_chopped) / np.sum(rho_induced_z_chopped)
print('The z-location of the electrical centroid is %.3f' % (centroid_z))

