#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:02:38 2020

@author: jiuqiwang
"""


from fit_gmm import fit,find_limits
from optimize_lpv_ds import optimize
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import data_loading as data_loading_module

class lpvds:
    
    '''
    attributes
    gmm as a gaussian mixture model object
    As as a model parameter
    bs as a model parameter
    k as the number of components of the gmm
    '''
    def __init__(self):
        self.gmm = None
        self.As = None
        self.bs = None
        self.k = 0
        
    '''
    input: trajectory in shape (sample size, dimension), 
           velocity in shape (sample size, dimension),tol indicating the tolerance,
           title as a String to name the model.
    output: None
    fit the model to represent the dynamical system
    '''
    def fit_ds(self, trajectory, velocity, tol = 0.00001, show_plot = False,show_lpvds = False,title = None):
   	
        print("Fitting GMM...")
        self.gmm = fit(trajectory,show_plot)
        print("GMM fitting complete.")
        
        print("Optimization start...")
        parameters = optimize(self.gmm,trajectory,velocity,tol)
        self.As = parameters[0]
        self.bs = parameters[1]
        self.k = self.As.shape[0]
        if show_lpvds:
            show_DS(self,trajectory,title)
       
        
    '''
    input: trajectory in shape (sample size, dimension), plot as a boolean variable
           to switch the result plotting on and off, scale factor as an float to
           scale down the size of the resultant velocity when plotting, proportion as
           an integer to indicate the portion of data points to plot, stability_check
           as an boolean value to switch the Lyapunov Stability checking on and off,
           stability_tol as an float to indicate the tolerance when performing the 
           stability checking
    output: estimated velocities in shape (sample size, dimension)
    '''
    def predict(self,trajectory,plot = False,scale_factor = 5, proportion = 10, stability_check = False, stability_tol = 0.1):
        
        # get the posterior probabilities
        posteriors = self.gmm.predict_proba(trajectory)
        
        # get the dimension
        dimension = trajectory.shape[1]
        
        # store the resultant vectors
        result = []
        
        # define the attractor
        attr = np.zeros((dimension,1))
        
        # get the estimated results
        for i in range(len(trajectory)):
            x = trajectory[i].reshape(dimension,1)
            weights = posteriors[i]
            result.append(estimate(x,self.As,self.bs,weights,self.k,dimension))
            
        result = np.array(result)
        
        # check stability if True
        if stability_check:
            
            # initialize the counter
            unstable = np.zeros(dimension)
            
            # count the data points that violate the Lyapunov Stability conditions
            for k in range(len(trajectory)):
                x_ref = trajectory[k].reshape(dimension,1)
                x_dot = result[k].reshape(dimension,1)
                unstable += check_stability(x_ref,x_dot, attr, stability_tol)
            
            # print the counted result
            print(str(int(unstable[0])) + ' data points violate the first Lyapunov Stability condition.')
            print(str(int(unstable[1])) + ' data points violate the second Lyapunov Stability condition.')

        # plot the result if True
        if(plot):
            # plot the predicted velocities
            plot_vectors(trajectory,result,scale_factor,proportion)
        
        return result
    
    
    
    
        
'''
input: clean trajectory in shape (sample size, dimension), clearn velocity 
       in shape (sample size, dimension), noisy_trajectory in shape (sample size,dimension),
       noisy_velocity in shape (sample size, dimension), num_of_splits as an integer, 
       tol indicating the tolerance
output: mean squared error and standard deviation in a tuple
'''
def cross_validate(clean_trajectory,clean_velocity,noisy_trajectory,noisy_velocity,num_of_splits = 10, tol = 0.00001):
    
    rs = ShuffleSplit(num_of_splits)
        
    # The sum of mse for all the folds
    error = []
    
    # Initialize a lpvds object
    model = lpvds()
        
    for train_index,test_index in rs.split(clean_trajectory):
            
        tra_train = noisy_trajectory[train_index]
        tra_test = clean_trajectory[test_index]
        vel_train = noisy_velocity[train_index]
        vel_test = clean_velocity[test_index]
            
        # fit the model
        model.fit_ds (tra_train,vel_train,tol)
            
        # get the result of prediction
        vel_predicted = model.predict(tra_test)
            
        error.append(mse(vel_predicted,vel_test))
    
    # calculate the mean and standard deviation
    error = np.array(error)
    mean = np.mean(error)
    std = np.std(error)
            
    return (mean,std)
    


'''
input: x as a vector, As as an array of matrices, bs as an array of vectors, weights as
       an array of scalars, k indicating the number of components, dimension as an integer
output: The estimated velocity in shape (dimension,)
'''
def estimate(x,As,bs,weights,k,dimension):
    vel = np.zeros(x.shape)
    for i in range(k):
        vel += weights[i]*(As[i].dot(x) + bs[i])
    return vel.reshape(dimension,)


    
'''
input: predicted velocities as an array of vectors, test set velocities as an array of vectors
output: a scalar as the mean squared error
'''    
def mse(predicted,label):
    diff = predicted-label
    n = len(predicted)
    cost = 0
    for v in diff:
        cost += v.dot(v.T)
    return cost/n



    
    
'''
input: trajectory in shape (sample size, dimension), velocity in shape (sample size, dimension),
       scale factor as an float, proportion as an integer
output: None
Plot the estimated velocities on the trajectory
'''
def plot_vectors(trajectory,velocity,scale_factor,proportion):
    dimension = trajectory.shape[1]
    
    if dimension<3:
        # caliberate the axis
        x_min,x_max,y_min,y_max = find_limits(trajectory)
        
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([x_min,x_max])
        axes.set_ylim([y_min,y_max])
        plt.grid()
        
        # plot the trajectory
        x = trajectory[:,0]
        y = trajectory[:,1]
        plt.scatter(x,y,color='green',marker = 'o',s = 5)
        
        # plot the vectors
        for i in range(len(trajectory)):
            norm = np.linalg.norm(velocity[i])
            scale = scale_factor*norm
            if (i%proportion==0):
                plt.arrow(trajectory[i][0],trajectory[i][1],
                          velocity[i][0]/scale,velocity[i][1]/scale,color = 'red',head_width = 0.05)
                
                
        plt.title('Predicted Vectors with Normalization Based on the Learned LPV-DS')
        plt.xlabel('x1')
        plt.ylabel('x2')
    else:
        # 3D plotting
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        plt.grid()
        # plot the trjectories
        X = trajectory[:,0]
        Y = trajectory[:,1]
        Z = trajectory[:,2]
        axes.scatter3D(X,Y,Z,c = Z,s = 5)
        
        # plot the vectors
        for i in range(len(trajectory)):
            norm = np.linalg.norm(velocity[i])
            scale = scale_factor*norm
            if (i%proportion == 0):
                axes.quiver(trajectory[i][0], trajectory[i][1], trajectory[i][2],
                            velocity[i][0]/scale, velocity[i][1]/scale, velocity[i][2]/scale, 
                            length=1,color = 'red',arrow_length_ratio=0.03)
                
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
        plt.title('Predicted Vectors with Normalization Based on the Learned LPV-DS')
        
    plt.show()



'''
input: position as a vector, attractor as a vector
output: a scalar value
'''
def V(position,attractor):
    return ((position-attractor).T).dot(position-attractor).item()



'''
input: position as a vector, velocity as a vector, attractor as a vector
output: a scalar
'''
def V_dot(position,velocity,attractor):
    return ((position-attractor).T).dot(velocity).item()


'''
input: position as a vector, velocity as a vector, attractor as a vector, tol as a float indicating the tolerance
output: an array of size (2,) indicating the number of data points that violate the first and second Lyapunov
        stability conditions respectively
'''
def check_stability(position,velocity,attractor,tol):
    stable = np.zeros(2)
    first = V(position,attractor)
    second = V_dot(position, velocity, attractor)
    if first < -tol:
        stable[0] = 1
    
    if second > tol:
        stable[1] = 1
        
    return stable


'''
input: ds as a lpvds object, trajectory in shape (sample size, dimension), title as a String
       to indicate the name of the model.
output: None
Plot the dynamical system based on the generated grid data
'''
def show_DS(ds,trajectory,title = None):
    dimension = trajectory.shape[1]
    if dimension<3:
        x_min,x_max,y_min,y_max = find_limits(trajectory)
        # calibrate the axis
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([x_min-1,x_max+1])
        axes.set_ylim([y_min-1,y_max+1])
        
        # plot the trajectory
        plt.grid()
        x = trajectory[:,0]
        y = trajectory[:,1]
        plt.scatter(x,y,color='green',marker = 'o',s = 5)
        
        # generate the grid data
        x_interval = np.linspace(x_min-1,x_max+1,80)
        y_interval = np.linspace(y_min-1,y_max+1,80)
        coordinate = []
        for x in x_interval:
            for y in y_interval:
                coordinate.append([x,y])
        coordinate = np.array(coordinate)
        
        # plot the estimated velocities
        res = ds.predict(coordinate)
        print("res",res.shape)
        for i in range(len(coordinate)):
            norm = np.linalg.norm(res[i])
            scale = 5*norm
            plt.arrow(coordinate[i][0],coordinate[i][1],res[i][0]/scale,
                      res[i][1]/scale,head_width=0.01,color='red')
            
        plt.xlabel('x1')
        plt.ylabel('x2')
    else:
        # 3D plotting
        x_min,x_max,y_min,y_max,z_min,z_max = find_limits(trajectory)
        
        # calibrate the axis
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        axes.set_xlim([x_min,x_max])
        axes.set_ylim([y_min,y_max])
        axes.set_zlim([z_min,z_max])
        
        # plot the trajectory
        plt.grid()
        X = trajectory[:,0]
        Y = trajectory[:,1]
        Z = trajectory[:,2]
        axes.scatter3D(X,Y,Z,c = Z,s = 5)
        
        # generate the grid data
        x_interval = np.linspace(x_min,x_max,6)
        y_interval = np.linspace(y_min,y_max,6)
        z_interval = np.linspace(z_min,z_max,6)
        coordinate = []
        for x in x_interval:
            for y in y_interval:
                for z in z_interval:
                    coordinate.append([x,y,z])
        coordinate = np.array(coordinate)
        
        # plot the estimated velocities
        res = ds.predict(coordinate)
        for i in range(len(coordinate)):
            norm = np.linalg.norm(res[i])
            scale = 3*norm
            axes.quiver(coordinate[i][0], coordinate[i][1], coordinate[i][2],
                        res[i][0]/scale, res[i][1]/scale, res[i][2]/scale, 
                        length=1,color = 'red')
            
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
    
    if title is None:
        plt.title('The Learned LPV-DS')
    else:
        plt.title('The Learned LPV-DS of ' + title)
        
    plt.show()










