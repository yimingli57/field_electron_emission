#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:04:36 2023

@author: apple
"""

# Find the average repulsion distance using the selected comparison method
# In this case, we optimize E=0 and fix E!=0. There are four different E-fields
# to compute, and the true repulsion distance is taken as the average of the 
# four computed values

import numpy as np

# Location of JDFTx output files
directory = '/Users/Apple/Desktop/MNED/Data/Au_100/'
file_names = ['Ecut_test_2/Au_100_E0_Ecut50.n',
              'E_test/Au_100_100MV_fix.n',
              'E_test/Au_100_300MV_fix.n',
              'E_test/Au_100_700MV_fix.n',
              'E_test/Au_100_1GV_fix.n']

# JDFTx parameters
fftbox = [56, 56, 1568]
e_fields = [0, 1e8, 3e8, 7e8, 1e9] # Applied electric field of each file
z_dim = 154.2

# Visualization setup
z_min = 0
z_max = 0.5

z_ind_min = int(z_min * fftbox[2])
z_ind_max = int(z_max * fftbox[2])

# Import the electron densities
rho_arr = np.empty((np.prod(fftbox),len(file_names)), dtype=float)
for i in range(len(file_names)):
    rho_arr[:,i] = np.fromfile(directory + file_names[i], dtype = np.float64)
rho_arr = np.reshape(rho_arr,(fftbox + [len(file_names)]))

# Find the induced electron densities
rho_induced = np.empty((fftbox + [len(file_names)-1]), dtype=float)
for i in range(len(file_names)-1):
    rho_induced[:,:,:,i] = rho_arr[:,:,:,i+1] - rho_arr[:,:,:,0]
rho_induced = rho_induced[:,:,z_ind_min:z_ind_max,:]

# Calculate the average position of the electrical surface
z_arr = np.linspace(z_min, z_max, num = (z_ind_max - z_ind_min)) * z_dim
rho_induced_z = np.mean(rho_induced,axis=(0,1)) # Average over the xy-plane
dz = z_arr[1] - z_arr[0] # Integration step
centroid_z_arr = np.empty(len(file_names)-1,dtype=float)
for i in range(len(centroid_z_arr)):
    centroid_z_arr[i] = np.sum(rho_induced_z[:,i]*z_arr) / np.sum(rho_induced_z[:,i])
centroid_z_avg = np.mean(centroid_z_arr)
print('The average location of the electrical surface is %.2f a.u.' % (centroid_z_avg))

#%% Save the average location to a text file
import pickle
with open('avg_elec_surf.txt', 'wb') as f:
    pickle.dump(centroid_z_avg,f)