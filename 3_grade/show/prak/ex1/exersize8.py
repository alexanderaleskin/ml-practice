# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 22:06:16 2015

@author: alex
"""

import numpy as np
from scipy.stats import multivariate_normal 
#from scipy.stats.norm import logpdf

def var_vec(X, m, C):
    """
    vectorized variant
    
    Parameters:
    X - vector of points of some size
    m - vector of mean
    C - matrix of covariance
    return log of solid normal distribution
    
    """
    
    Ex = np.exp(np.sum((X - m).\
         dot(np.linalg.inv(C))*(X - m), axis = 1) * (-0.5))
    alfa = 1 / (((2*np.pi)**C.shape[0] * np.linalg.det(C)) ** 0.5)
    return np.log(Ex * alfa)
    
def var_non_vec(X, m, C):
    """
    non vectorized variant
    
    Parameters:
    X - vector of points of some size
    m - vector of mean
    C - matrix of covariance
    
    """
    ret_vec = np.empty((X.shape[0]))
    alfa = 1 / (((2*np.pi)**C.shape[0] * np.linalg.det(C)) ** 0.5)
    inv_C = np.linalg.inv(C)
    for i in range(X.shape[0]):
        var_vec = np.zeros((X.shape[1]))
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                var_vec[j] += (X[i, k] - m[k]) * inv_C[k, j]
            var_vec[j] *= (X[i,j] - m[j])
        summ = 0.0
        for j in var_vec:
            summ += j
        ret_vec[i] = np.log(np.exp(summ * (-0.5))* alfa)
    return ret_vec
    
def var_extra(X, m, C):
    """
    using scipy.stats.multivariate_normal
    and scipy.stats.norm.logpdf
    
    Parameters:
    X - vector of points of some size
    m - vector of mean
    C - matrix of covariance
    
    """
    return multivariate_normal(m, C).logpdf(X)
    