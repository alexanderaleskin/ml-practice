# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:18:58 2015

@author: alex
"""

import numpy as np

def var_vec(x):
    """
    vectorized variant
    x - vector of data
    return max elem before which there is 0.    
    
    """  
    maska = np.roll(x == 0, 1)
    maska[0] = False
    return np.amax(x[maska])
    
def var_non_vec(x):
    """
    non vectorized variant
    x - vector of data
    return max elem before which there is 0.  
    
    """  
    val_max = 0
    for i in range(1, x.size):
        if (x[i-1] == 0) & (x[i] > val_max):
            val_max = x[i]
    return val_max
    
def var_extra(x):
    """
    combo of vectorized and not vectorized
    x - vector of data
    return max elem before which there is 0.  
    
    """
    val_max = 0
    try:            
        for i in np.where(x == 0)[0]:
            if x[i+1] > val_max:
                val_max = x[i+1]
    except IndexError:
        pass
    return val_max
    