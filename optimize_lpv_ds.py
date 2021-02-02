#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:21:43 2020

@author: jiuqiwang
"""
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag
import numpy as np


'''
input: an array of flattened parameters A and b, k as an integer indicating the number of
        components of the gmm, trajectory data in shape (sample size, dimension, 1),
        velocity data in shape (sample size, dimension, 1), dimension as an integer and
        a list of posterior probabilities regarding to the trajectory data points
        
output: a scalar as a float indicating the cost of the model
This defines the objective function that needs to be minimized
'''
def objective(parameter,k,trajectory,velocity,dimension,posterior):
    
    # restore the parameters
    As,bs = restore(parameter,dimension,k)
    components = range(k)
    errors = np.empty(velocity.shape)
    for num,x in enumerate(trajectory):
        
        # get posterior probability
        weights = posterior[num]
        
        # initialize the estimated velocity as zero
        est_vel = np.zeros((dimension,1))
        
        # get the estimated velocity
        for i in components:
             est_vel = est_vel + weights[i]*(As[i].dot(x)+bs[i])
             
        # get the reference vector
        ref_vel = velocity[num]
        
        # calculate and add the error vector to the array
        errors[num] = est_vel - ref_vel
        
    return np.linalg.norm(errors)


'''
input: As and bs as arrays of matrices and vectors respectively
output: a flattened and concatenated 1-D array of As and bs
'''
def smash(As,bs):
    return np.concatenate((As.flatten(),bs.flatten()))


'''
input: an array of parameters of A and b, dimension as an integer, k as an integer indicating
       the number of components of the gmm
output: a tuple of parameters A and b in matrix and vector form 
'''
def restore(parameter,dimension,k):
    
    #split the array into the A part and b part
    As = parameter[:dimension*dimension*k]
    bs = parameter[dimension*dimension*k:]
    
    #reshape them into the original form
    As = As.reshape(k,dimension,dimension)
    bs = bs.reshape(k,dimension,1)
    return (As,bs)
        

'''
input: dimension as integer, k as integer indicating the number of components
output: a matrix used to define the linear constraint
'''
def transformation_matrix(dimension,k):
    
    # transpose matrix for one A matrix
    transform = np.zeros((dimension*dimension,dimension*dimension))
    for i in range(dimension):
        for j in range(dimension):
            row = i*dimension + j
            column = j*dimension + i
            transform[row,column] = 1
            
    # identity matrix for one A matrix
    A_ident = np.identity(dimension*dimension)
    
    # combination of two linear transformations
    transform = transform+A_ident
    
    # stack the matrices diagonally for the k components
    stack_transform = transform
    for i in range(k-1):
        stack_transform = block_diag(stack_transform,transform)
    
    # get the length for b part
    b_length = dimension*k
    # identity matrix for b part
    b_ident = np.identity(b_length)
    
    # stack the two parts together
    stack_transform = block_diag(stack_transform,b_ident)
    return stack_transform


'''
input: dimension as an integer, k as an integer indicating the number of components of the gmm,
       tol as a float indicating the tolerance
output: a linear constraint for optimization
'''
def get_constraints(dimension,k,tol):
    
    trans_matrix = transformation_matrix(dimension, k)
    # A part length
    A_length = dimension*dimension*k
    # b part length
    b_length = dimension*k
    
    # lower bound: A matrices : negative infinity  b vectors: -tol (close to zero)
    A_lb = np.full((A_length),-np.inf)
    b_lb = np.full((b_length),-tol)
    lb = np.concatenate((A_lb,b_lb))
    
    # upper bound: A matrices & b vectors: tol (close to zero)   
    ub = np.full((A_length+b_length),tol)
    
    #form the linear constraint
    linear_constraint = LinearConstraint(trans_matrix, lb, ub)
    return linear_constraint


'''
input: a gaussian mixture model gmm, trajectory in shape (sample size, dimension), 
       velocity in shape (sample size, dimension), tol as a float indicating tolerance,
       init as the initial parameters
output: a tuple of parameters (As, bs) after optimization
'''
def optimize(gmm,trajectory,velocity,tol,init = None):
    
    # number of components of the gmm
    k = len(gmm.weights_)
    # size and dimension of data
    size,dimension = trajectory.shape
    # posterior of the reference trajectory
    posterior = gmm.predict_proba(trajectory)
    # reshape the trajectory
    trajectory = trajectory.reshape(size,dimension,1)
    # reshape the velocity
    velocity = velocity.reshape(size,dimension,1)
    
    if init is None:
        # random initialization of parameters
        init = np.random.rand(dimension*dimension*k + dimension*k)
    
    # get the constraint for this problem
    cons = get_constraints(dimension,k,tol)
    
    # get the result of the optimization
    res = minimize(objective,init,(k,trajectory,velocity,dimension,posterior),constraints = cons)
    print(res.message)
    
    #restore the parameter
    restored = restore(res.x,dimension,k)
    return restored


