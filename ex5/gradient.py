# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:17:57 2016

@author: alex
"""

def compute_gradient(J, theta):   
    """
    Compute gradient by the difference quotient.
    
    J - mathematic function
    theta - point in which we look gradient (ndarray)
    
    """
    
    import numpy as np
    eps = 1e-5
    
    if type(theta) != np.ndarray or theta.ndim != 1:
        raise TypeError('theta must be ndarray 1-dimensioanl')
        
    ans = np.empty(theta.shape)
    for i in range(theta.shape[0]):
        e = np.zeros_like(ans)
        e[i] = eps 
#        if i % 1000 == 0:
#            print(i)
        ans[i] = (J(theta + e) - J(theta - e)) / (2 * eps)
    return ans
    
def check_gradient():
    """
    Check function compute_gradient
    """    
    
    import numpy as np    
    # x is ndarray
    def linear(x):
        return 2 * x[0] + 0.5
        
    def grad_linear(x):
        return 2        
        
    def square(x):
        return 5 * x[0] ** 2 - 0.22
    
    def grad_square(x):
        return 10 * x[0]
    
    def cube(x):
        return x[0] ** 2 + 2 * x[1] ** 3 + 4 * x[2] ** 3 + 0.5
        
    def grad_cube(x):
        return np.array([2 * x[0], 6 * x[1] ** 2, 12 * x[2] ** 2])        
        
    def f1(x):
        return np.sqrt(x[0]) + np.sin(x[1]) + np.log(x[2]) + np.exp(np.abs(x[3]) ** 1.5)
        
    def grad_f1(x):
        return np.array([0.5 * x[0] ** (-0.5), np.cos(x[1]), 1 / x[2],
                                         (1.5 * np.abs(x[3]) ** 0.5) * np.exp(np.abs(x[3]) ** 1.5)])
    
    check = [
                [linear, grad_linear, np.array([0]), np.array([50])],
                [square, grad_square, np.array([0.01]), np.array([-30])],
                [cube, grad_cube, np.array([0, 0.1, -0.01]), np.array([10, 20, 30])],
                [f1, grad_f1, np.array([0.01, np.pi/6, 1, 0]), np.array([100, np.pi / 2, 0.01, 5])]             
            ]
    
    results = np.zeros(len(check), dtype = int)
    for i in range(len(check)):
        f, grad, p1, p2 = check[i]
        if np.isclose(compute_gradient(f, p1), grad(p1), atol = 1e-6).all() and \
            np.isclose(compute_gradient(f, p2), grad(p2), atol = 1e-6).all():
            results[i] = 1
        else:
            results[i] = 0
    if np.all(results):
        print('all right')
    else:
        print('results: ', results)
    return