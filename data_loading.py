#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:48:15 2020

@author: jiuqiwang
"""
from scipy.io import loadmat;
import numpy as np


'''
input: positional data in shape (dimension, sample size)

output: shifted positional data in shape (dimension, sample size) where 
the target is (0,0)
'''
def calibrate(pos):
    
    # each dimension is shifted so that the last data point ends in the origin
    for i in range(len(pos)):
        pos[i] = pos[i] - pos[i,-1]
    
    return pos


'''
input: The file name in string of the dataset
output: The processed trajectory and velocity data in a tuple (trajectory, velocity)
where the shape of trajectory data and velocity data is (sample size, dimension)
'''
def load(filename):
    
    # load the raw data
    raw = loadmat('dataset/' + filename + '.mat')
    # strip out the data portion
    data = raw["data"].reshape(-1)
    # calculate dimension of the data
    dimension = data[0].shape[0]//2
    
    # strip out the trajectory and velocity portions
    pos_list = []
    vel_list = []
    for demonstration in data:
        pos = demonstration[0:dimension]
        vel = demonstration[dimension:]
        pos_calibrated = calibrate(pos)
        pos_list.append(pos_calibrated)
        vel_list.append(vel)
    
    # concatenate the results
    concatenated_pos = np.concatenate(pos_list,axis = 1)
    concatenated_vel = np.concatenate(vel_list,axis = 1)
    
    return (concatenated_pos.T,concatenated_vel.T)
