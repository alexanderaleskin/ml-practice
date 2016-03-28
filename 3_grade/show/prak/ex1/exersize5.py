# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:12:21 2015

@author: alex
"""
import numpy as np

def var_vec(Image, numChannels = np.array([0.299, 0.587, 0.114])):
    """
    vectorized variant
    Image - array height* width* numChannels size
    numChannels - vector with weight of color
    
    """
    return np.uint8(np.round(np.sum(Image*numChannels, axis = 2)))
    
def var_non_vec(Image, numChannels = np.array([0.299, 0.587, 0.114])):
    """
    not vectorized variant
    Image - array height* width* numChannels size
    numChannels - vector with weight of color
    
    """
    black = np.zeros(Image.shape[0:2], dtype= Image.dtype)
    for i in np.arange(Image.shape[0]):
        for j in np.arange(Image.shape[1]):
            val  = 0.0
            for k in range(3):   
                val += Image[i, j, k] * numChannels[k]
            black[i, j] =np.uint8(round(val))
    return black
    
def var_extra(Image, numChannels = np.array([0.299, 0.587, 0.114])):
    """
    variant there is no another variant.
    so it is something between 2 first variants
    Image - array height* width* numChannels size
    numChannels - vector with weight of color
    
    """
    black = np.zeros(Image.shape[0:2])
    for k in range(3):
        black += Image[:,:,k]*numChannels[k]
    return np.uint8(np.round(black))
    
    