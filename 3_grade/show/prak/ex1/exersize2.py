# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:02:04 2015

@author: alex
"""
import numpy as np

def var_vec(X, i, j):
    """ 
    vectorized variant
    X - matrix of values
    i, j - vectors of indexes with same length.
    return np.array with elements in [i, j] position in X.
    
    """
    return X[i,j].copy()
    
def var_non_vec(X, i, j):
    """ 
    non vector variant
    X - matrix of values
    i, j - vectors of indexes with same length.
    return np.array with elements in [i, j] position in X.
    
    """
    Y = []
    for k in range(len(i)):
        Y += [X[i[k], j[k]]]
    return np.array(Y)
    
def var_extra(X, i, j):
    """
    variant with list comprehesion
    X - matrix of values
    i, j - vectors of indexes with same length.
    return np.array with elements in [i, j] position in X.
    
    """
    return np.array([X[i[k],j[k]] for k in range(len(i))])
    