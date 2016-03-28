# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 23:50:04 2016

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

def display_layer(X, filename):
    """
    X -  ndarray N * D, where N - amount of image, D - size of image with type
    3 * d * d. For show image in ipython use filename = 1
    
    
    Return subplot with the biggest amount of pictures with square form
    
    """
    
    if type(X) != np.ndarray:
        raise TypeError('images must be of numpy.ndarray')
    if X.ndim != 2:
        raise TypeError('images is 2 dimensional')
    
    sz = np.sqrt(X.shape[1] / 3)
    if np.isclose(int(sz), sz):
        shape = (int(sz), int(sz), 3)
    else:
        raise ValueError('length of iamge should be 3 *row^2')
                
    n_row_im = np.floor(np.sqrt(X.shape[0]))
    all_im = int(n_row_im ** 2)
    plot = plt.figure(figsize = (7, 7))
    for i in range(all_im):
        image = X[i].reshape(shape)
        ad = plot.add_subplot(n_row_im, n_row_im, i+1) # указываем номер изображения
        ad.imshow(image, interpolation='none')
        ad.axis("off")
    plot.savefig(filename)
    return