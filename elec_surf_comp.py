#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:07:42 2023

@author: apple
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('~/Desktop/MNED/Data/Au_100/elec_surf_compare.csv',delimiter=',')

# Ionic positions are optimized at E = 0, and fixed at E != 0
data_E0_opt = data[(data.loc[:,'E=0_fix'] == 0) & (data.loc[:,'E!=0_fix'] == 1)]
# Ionic positions are fixed at E = 0, and optimized at E != 0
data_E0_fix = data[(data.loc[:,'E=0_fix'] == 1) & (data.loc[:,'E!=0_fix'] == 0)]
# Ionic positions are optimized at both E = 0 and E != 0
data_both_opt = data[(data.loc[:,'E=0_fix'] == 0) & (data.loc[:,'E!=0_fix'] == 0)]
# Electric fields in GV/m
E_fields = data_E0_opt.loc[:,'E-field']/1e9

#%% Plotting
fig1,ax1 = plt.subplots()
ax1.plot(E_fields, data_E0_opt.loc[:,'L2_opt'], label='E=0 opt, L2 min', 
         marker='.', markersize=12, linestyle='--')
ax1.plot(E_fields, data_E0_opt.loc[:,'centroid'], label='E=0 opt, centroid', 
         marker='.', markersize=12, linestyle='--')
ax1.plot(E_fields, data_E0_fix.loc[:,'L2_opt'], label='E=0 fix, L2 min',
         marker='.', markersize=12, linestyle='--')
ax1.plot(E_fields, data_E0_fix.loc[:,'centroid'], label='E=0 fix, centroid',
         marker='.', markersize=12, linestyle='--')
ax1.plot(E_fields, data_both_opt.loc[:,'L2_opt'], label='both opt, L2 min',
         marker='.', markersize=12, linestyle='--')
ax1.legend()
ax1.set_xlim([0,1.1])
ax1.set_xlabel('E-field (GV/m)')
ax1.set_ylabel('Electrical Surface Location (Bohr)')
ax1.set_title('Electrical Surface Locations from 5 Different Models')

#%% Take the average of the electrical surface locations using E=0 opt, centroid rule
elec_surf_arr = data_E0_opt.loc[:,'centroid'].to_numpy()
elec_surf_loc = np.mean(elec_surf_arr)
print('The average location of the electrical surface is found at %.3f Bohr' % (elec_surf_loc))