# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:54:33 2015

@author: alex
"""
import numpy as np

def var_vec(x,y):
    """
    vectorized variant
    x, y - vectors(multisets) of type int
    return true if multisets are the same
    
    """
    val = False
    if x.size == y.size:
        x_val, x_count = np.unique(x, return_counts = True)
        y_val, y_count = np.unique(y, return_counts = True)
        if len(x_val) == len(y_val):        
            val = (x_val == y_val).all()
            if val:
                val &= (x_count == y_count).all()
    return val
    
def var_non_vec(x,y):
    """
    non vectorized variant
    x, y - vectors(multisets) of type int
    return true if multisets are the same
    
    """   
    val = False
    if x.size == y.size:
        x_val = []
        y_val = []
        x_count = []
        y_count = []
        for i in range(x.size):
            for j in range(len(x_val)):
                if x[i] == x_val[j]:
                    x_count[j] += 1
                    break
            else:
                x_val += [x[i]]
                x_count += [1]
            for j in range(len(y_val)):
                if y[i] == y_val[j]:
                    y_count[j] += 1
                    break
            else:
                y_count += [1]
                y_val += [y[i]]
        if len(x_val) == len(y_val):
            val = True
            for i in range(len(x_val)):
                for j in range(len(y_val)):
                    if y_val[j] == x_val[i]:
                        break
                else:
                    val = False
                    break
                if y_count[j] != x_count[i]:
                    val = False
                    break  
    return val
                
def var_extra(x,y):
    """
    this variant doing sorting and equal sorted vectors
    x, y - vectors(multisets) of type int
    return true if multisets are the same
    
    """   
    val = False
    if x.size == y.size:
        a = x.copy()
        b = y.copy()
        a.sort()
        b.sort()
        if a.size == b .size:            
            if (a == b).all() :
                val = True
    return val    
    
def var_extra2(x,y):
    """
    this variant using bincount
    x, y - vectors
    return true if multisets are the same
    
    """  
    val = False
    if x.size == y.size:
        x_b = np.bincount(x) 
        y_b = np.bincount(y)
        if (x_b.size == y_b.size):
            if (x_b == y_b).all():
                val = True
    return val