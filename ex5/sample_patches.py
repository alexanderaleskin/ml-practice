# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def normalize_data(images):
    """
    Return normalize values: do сutting over 3*dispersion and normalize in [0.1, 0.9]
    
    images is numpy.ndarray With shape N*D, N- amount of images, D- size of it
    
    """
    
    if type(images) != np.ndarray:
        raise TypeError('images must be of numpy.ndarray')
    
    stds = 3 * np.std(images, axis = 0)
    means = np.mean(images, axis = 0)
    cut_images = np.minimum(images, np.repeat(means[np.newaxis], images.shape[0], 0) + stds)
    cut_images = np.maximum(cut_images, np.repeat(means[np.newaxis], images.shape[0], 0) - stds)
    cut = cut_images - means
    min1 = np.min(cut)
    max1 = np.max(cut)
    if -min1 > max1:
        max1 = -min1
    return cut / max1 / 2.5 + 0.5
    
    
def sample_patches_raw(images, num_patches=10000, patch_size=8):
    """
    Return  array N * D, N - number of patches, D - size of putch - 3 * patch_size ^ 2;
    
    """
    
    if type(images) != np.ndarray:
        raise TypeError('images must be of numpy.ndarray')
    if images.ndim != 2:
        raise TypeError('images is 2 dimensional')
    
    sz = np.sqrt(images.shape[1] / 3)
    if np.isclose(int(sz), sz):
        shape = (int(sz), int(sz), 3)
    else:
        raise ValueError('length of iamge should be 3 *row^2')
    
    num_im = np.random.randint(0, images.shape[0], num_patches)
    coord = np.random.randint(0, shape[0] - patch_size, num_patches * 2)
    patches = np.empty((num_patches, 3 * patch_size ** 2), dtype = images.dtype)    
    for i in range(num_patches):
        image = images[num_im[i]].reshape(shape)
        patches[i] = image[coord[i * 2] : coord[i * 2]+ patch_size,
                      coord[i * 2 + 1]: coord[i * 2 + 1] + patch_size, :].copy().reshape(-1)
    return patches
    
def sample_patches(images, num_patches=10000, patch_size=8):
    patches = sample_patches_raw(images, num_patches, patch_size)
    return normalize_data(patches)