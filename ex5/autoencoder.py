# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:09:38 2016

@author: alex
"""

def initialize(hidden_size, visible_size):
    import numpy as np    
    """
    hidden_size - symmetric 1D ndarray, each element is amount of neurons in each level
    visible_size - amount of neurons in visible level (input/output)    
    
    """
    
    if type(hidden_size) != np.ndarray:
        raise TypeError('hidden_size must be of numpy.ndarray')
    if hidden_size.ndim != 1:
        raise TypeError('hidden_size is 1 dimensional')
    if not (hidden_size[::-1] == hidden_size).all():
        raise TypeError('hidden_size must be symmetric')
        
    vec = np.concatenate((np.array([visible_size]), hidden_size, np.array([visible_size])))
    arr = np.array([])
    
    for i in range(len(vec) - 1):
        edge = np.sqrt(6 / (vec[i] + vec[i + 1] + 1))
        rand = np.random.uniform(- edge, edge, vec[i]* vec[i + 1])
        arr = np.concatenate((arr, rand, np.zeros(vec[i + 1])))
    return arr
    
def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    
    import numpy as np 

    sizes = np.concatenate((np.array([visible_size]), hidden_size, np.array([visible_size])))
    offset = 0
    n = data.shape[0]
    J = 0
    Z = [data]
    alfa = lambda_ 
    p = sparsity_param
    p_real2 = []
    
    for i in range(len(sizes) - 1):
        W = theta[offset : offset + sizes[i] * sizes[i + 1]]
        W = W.reshape(sizes[i], sizes[i + 1])
        b = theta[offset + sizes[i] * sizes[i + 1] : offset + (sizes[i] + 1) * sizes[i + 1]]
        offset += (sizes[i] + 1) * sizes[i + 1]
        A = Z[-1].dot(W) + b
        Z.append(1 / (1 + np.exp(-A)))
        J += alfa / 2 * (np.sum(W ** 2))

    S = data - Z[-1]
    J += np.sum((S) ** 2) / (n * 2)
    dJ = np.array([])
    S = - S / n
    offset = theta.size
    

    for i in range(2, len(sizes) + 1):
        W = theta[offset -  (sizes[-i] + 1) * sizes[-i + 1] : offset - sizes[-i + 1]]
        W = W.reshape(sizes[-i], sizes[-i + 1])
        b = theta[offset - sizes[-i + 1] : offset]
        offset -= (sizes[-i] + 1) * sizes[-i + 1]
        
        Z_del = (1 - Z[-i + 1]) * Z[-i + 1] 
        S = S * Z_del
        dW = Z[-i].T.dot(S) + alfa * W
        db = np.sum(S, axis = 0)
        
        p_real = np.sum(Z[-i], axis = 0) / n
        if i > 2:
            J += beta * np.sum(p * np.log(p / p_real2) + (1 - p) * np.log((1 - p) / (1 - p_real2)))
            
        if i < len(sizes):
            p_real2 = p_real
            coefs = beta * ((1 - p) / (1 - p_real) - p / p_real) / n
            S = S.dot(W.T) + coefs
        
        dJ = np.concatenate((dW.reshape(-1), db.reshape(-1), dJ))

    return (J, dJ)

    
def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
    """
    to get result of autoencoder use layer_number with value -1 (defualt)
    to see result on i-layer use layer_number with appropiriete value i
    
    """
    import numpy as np    
    
    sizes = np.concatenate((np.array([visible_size]), hidden_size, np.array([visible_size])))
   
    if layer_number > sizes.size - 1: 
        raise ValueError('layer_number is bigger then amount of layers')
    elif layer_number == -1:
        layer_number = sizes.size - 1
        
    offset = 0
    Z = data
    
    for i in range(layer_number):
        W = theta[offset : offset + sizes[i] * sizes[i + 1]]
        W = W.reshape(sizes[i], sizes[i + 1])
        b = theta[offset + sizes[i] * sizes[i + 1] : offset + (sizes[i] + 1) * sizes[i + 1]]
        offset += (sizes[i] + 1) * sizes[i + 1]
        A = Z.dot(W) + b
        Z = 1 / (1 + np.exp(-A))

    return Z
    
