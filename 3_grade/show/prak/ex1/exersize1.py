# -*- coding: utf-8 -*-

import numpy as np


def var_vec(X):
    """
    vectorised variant
    
    X - np.array, X.dim == 2
    return product of multiplication of non-zero diagonal elements
    
    """
    diag = X.diagonal()
    return np.multiply.reduce(diag[diag != 0])

def var_non_vec(X):
    """
    non vectorised variant
    
    X - np.array, X.dim == 2
    return product of multiplication of non-zero diagonal elements
    
    """
    x = X.shape[0]
    y = X.shape[1]
    if x > y:
        min_sz = y
    else:
        min_sz = x
    sum_val = 1
    for i in range(min_sz):
        if X[i,i] != 0:
            sum_val = sum_val * int(X[i,i])
    return sum_val
    
def var_extra(X):
    """
    extra variant
    
    X - np.array, X.dim == 2
    return product of multiplication of non-zero diagonal elements
    
    """
    min_sz = min((X.shape))
    mass = [X[i,i] for i in range(min_sz) if X[i,i] != 0]
    return np.multiply.reduce(mass)
    
    
        