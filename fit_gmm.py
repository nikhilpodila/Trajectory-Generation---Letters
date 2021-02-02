#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:51:23 2020

@author: jiuqiwang
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from sklearn import mixture
from mpl_toolkits import mplot3d
'''
input: positional data in shape(sample size, dimension)
output: None
Plot the trajectory used for training the model
'''
def plot_trajectory(trajectory):
    dimension = trajectory.shape[1]
    plt.figure()
    if dimension < 3:
        X1 = trajectory[:,0]
        X2 = trajectory[:,1]
        plt.scatter(X1,X2,marker = 'o',s = 5)
        plt.xlabel("x1")
        plt.ylabel("x2")
    else:
        X = trajectory[:,0]
        Y = trajectory[:,1]
        Z = trajectory[:,2]
        ax = plt.axes(projection='3d')
        ax.scatter3D(X,Y,Z,c = Z,s = 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    plt.title("Reference Trajectories")
    plt.grid()
    plt.show()

    
    
'''
input: a list of floats
output: a list of floats representing the difference between each pair of consecutive
floats in the input list
'''
def diff(arr):
    difference = []
    for i in range(len(arr)-1):
        difference.append(arr[i+1]-arr[i])
    return difference




'''
input: bic scores as a list, number of components as an integer, plot as a boolean switch for plotting
output: the index of the best model based on the second derivative of the bic scores
        respect to the number of components
'''
def select_model(bics,num_components,plot):
    #calculate the first and second order derivative
    diff1 =[0] + diff(bics)
    diff2 = [0]+ diff(diff1)
    
    if plot:
        #plot the bic and diff1 and diff2
        fig,axs = plt.subplots(2)
        axs[0].plot(num_components,bics,label = "BIC")
        axs[0].set_title("BIC Score for GMM fit")
        axs[0].grid()
        axs[0].set_xlabel("Number of Gaussian Functions")
        axs[0].legend()
        
        axs[1].plot(num_components,diff1,label = "diff1(BIC)")
        axs[1].plot(num_components,diff2,label = "diff2(BIC)")
        axs[1].set_title("First and Second Derivative for GMM fit")
        axs[1].grid()
        axs[1].set_xlabel("Number of Gaussian Functions")
        axs[1].legend()
    
        plt.tight_layout()
        plt.show()
        
    return diff2.index(max(diff2))



'''
input: trajectory in shape (sample size, dimension)
output: integer tuple(x_min, x_max, y_min, y_max) or (x_min, x_max, y_min, y_max, z_min, z_max)
        representing the bounds of the trajectory data
'''
def find_limits(trajectory):
    dimension = trajectory.shape[1]
    # for dimension 2
    if dimension < 3:
        x_min,y_min = np.inf,np.inf
        x_max,y_max = -np.inf,-np.inf
        for point in trajectory:
            if point[0]<x_min:
                x_min = point[0]
            if point[0]>x_max:
                x_max = point[0]
            if point[1]<y_min:
                y_min = point[1]
            if point[1]>y_max:
                y_max = point[1]
        return (math.floor(x_min),math.ceil(x_max),
                math.floor(y_min),math.ceil(y_max))
    # for dimension 3
    else: 
        x_min,y_min,z_min = np.inf,np.inf,np.inf
        x_max,y_max,z_max = -np.inf,-np.inf,-np.inf
        for point in trajectory:
            if point[0]<x_min:
                x_min = point[0]
            if point[0]>x_max:
                x_max = point[0]
            if point[1]<y_min:
                y_min = point[1]
            if point[1]>y_max:
                y_max = point[1]
            if point[2]<z_min:
                z_min = point[2]
            if point[2]>z_max:
                z_max = point[2]
        return (math.floor(x_min),math.ceil(x_max),
                math.floor(y_min),math.ceil(y_max),
                math.floor(z_min),math.ceil(z_max))
   
    

'''
input: trajectory in shape (sample size, dimension), means as an array of coordinates, 
       covariances as an array of matrices
output: None
This function plots the covariance and mean of the components of the GMM on the reference
trajectory.
'''   
def plot_gmm(trajectory,means,covariances):
    dimension = trajectory.shape[1]
    if dimension < 3:
        # generate the elipses for gmm components
        ellipses = []
        for i in range(len(means)):
            v,w = np.linalg.eigh(covariances[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi
            e = mpl.patches.Ellipse(means[i], v[0], v[1], 180. + angle)
            ellipses.append(e)
            
        # plot the trajectory
        _,ax = plt.subplots()
        X1 = trajectory[:,0]
        X2 = trajectory[:,1]
        plt.scatter(X1,X2,marker = 'o',s = 5)
        
        # plot the means
        for mean in means:
            plt.plot([mean[0]],[mean[1]],marker = 'x',markersize = 8,color='red')
        
        # plot the ellipses
        for ell in ellipses:
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.6)
            ell.set_facecolor(np.random.rand(3))
        
        x_min,x_max,y_min,y_max = find_limits(trajectory)
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        plt.grid()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('GMM Components on Reference Trajectory')
        plt.show()


    
'''
input: trajectory data in shape (sample size, dimension)
output: a gaussian mixture model object
'''
def fit(trajectory,plot):
    
    if plot:
        #Plot the trajectory of the dataset
        plot_trajectory(trajectory)
    
    #store the bic scores and the corresponding GMMs
    bics = []
    gmms = []
    num_components = range(1,11)
    
    # fit the gmms
    for num in num_components:
        #fit the model
        gmm = mixture.GaussianMixture(n_components = num)
        gmm.fit(trajectory)
        gmms.append(gmm)
        #get bic score
        current_bic = gmm.bic(trajectory)
        bics.append(current_bic)
    
    # find the best model
    gmm = gmms[(select_model(bics,num_components,plot))]
    
    if plot:
        # plot the gmm components
        plot_gmm(trajectory,gmm.means_,gmm.covariances_)
        

    # return the best model
    return gmm
