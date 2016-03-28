# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 23:57:44 2015

@author: alex
"""

import numpy as np
import scipy.spatial.distance as dist

def var_vec(X, Y):
    """
    vectorized variant
    X,Y - objects variety of type array
    return array with dist between Xi and Yj objects in (i,j) element. 
    
    """
    XX = np.repeat(X, Y.shape[0], axis = 0)
    YY = np.tile(Y, (X.shape[0], 1))
    Z = np.sum((XX-YY)**2, axis =1)**(0.5)
    return np.reshape(Z, (X.shape[0], Y.shape[0]))     
    
    
def var_non_vec(X, Y):
    """
    not vectorized variant
    X,Y - objects variety of type array
    return array with dist between Xi and Yj objects in (i,j) element. 
    
    """
    Z = np.empty([X.shape[0], Y.shape[0]])
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            val = 0.0
            for k in range(X.shape[1]):
                val += (X[i, k] - Y[j, k])**2
            Z[i,j] = val**(0.5)
    return Z
            



def var_extra(X, Y):
    """
    using scipy.spatial.distance.cdist
    X,Y - objects variety of type array
    return array with dist between Xi and Yj objects in (i,j) element. 
    
    """
    return dist.cdist(X, Y)
    