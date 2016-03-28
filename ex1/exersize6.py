# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:25:24 2015

@author: alex
"""

import numpy as np

def var_vec(x):
    """
    vectorized variant
    x - enter vector of type narray
    return {numbers, counts} where numbers and counts are np.array
    
    """
    diff = x[1:] - x[:-1]
    pos = np.where(diff != 0)[0] +1
    reps = pos[1:] - pos[:-1]
    vals = np.empty((pos.shape[0]+1), dtype = x.dtype)
    vals[0] = x[0]
    vals[1:] = x[pos].copy()
    repeats = np.empty((pos.shape[0]+1), dtype = pos.dtype)
    repeats[0] = pos[0] - 0
    repeats[1:-1] = reps
    repeats[-1] = x.size - pos[-1]
    return (vals, repeats)


def var_non_vec(x):
    """
    not vectorized variant
    x - enter vector
    return {numbers, counts} where numbers and counts are np.array
    
    """
    val  = x[0];
    count = 0;
    val_list = [];
    count_list = [];
    for i in x:
        if i == val:
            count += 1
        else:
            val_list.append(val)
            count_list.append(count)
            count = 1
            val = i
    val_list.append(val)
    count_list.append(count)
    return (np.array(val_list), np.array(count_list))
    
def var_extra(x):
    """
    list comprehension variant
    same as var_extra
    x - enter vector
    return {numbers, counts} where numbers and counts are np.array
    
    """
    pos = [0] + [i for i in range(1,x.size) if x[i] != x[i-1]]
    counts = [pos[i] - pos[i-1] for i in range(1, len(pos))] +\
              [x.size - pos[-1]]
    return (x[np.array(pos)].copy(), np.array(counts))
    
    