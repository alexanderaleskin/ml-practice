# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

if you find mistakes please contact alexanderaleskin@mail.ru :)
"""

import cvxopt as cv
import numpy as np    
import scipy.spatial.distance as dist
import time
from sklearn import svm
import matplotlib.pyplot as plt
# import warnings

class MySVM:    
    def __init__(self):
        return 

    def fit(self, X, y, C):
        self.X_ = X
        try:        
            if y.shape[1] == 1:
                self.y_ = y
            else:
                raise TypeError
        except IndexError:
            self.y_ = y[np.newaxis].T
        self.C_ = C
        self.A_ = None
        self.w_ = None
        self.b_ = None
        self.ksi_ = None
        self.sup_vec_ = None
        self.gamma_ = None
        self.obj_curve_ = None
#        self.iter_ = None
        return
        
    def _make_svm_result_(self, w = None, A = None, status = None, time = None, 
                        obj_curve = None, b = None, ksi = None, iter_ = None):
        
        self.A_ = A                    
        if type(w) != np.ndarray  and type(A) == np.ndarray and self.gamma_ == 0:            
            w = self.compute_w()
        self.w_ = w
        self.b_ = b
        self.ksi_ = ksi
        self.sup_vec_ = None
        if obj_curve is None:
            if type(A) == np.ndarray:
                obj_curve = self.compute_dual_objective()
            elif type(self.w_) == np.ndarray:
                obj_curve = self.compute_primal_objective()
            else:
                raise TypeError
        self.obj_curve_ = obj_curve  
#        self.iter_ = iter_
        return {'w': w,
                'b': b,
                'iter': iter_,
                'A': A,
                'status': status,
                'time': time,
                'objective_curve': obj_curve
               }
    
    def _base_(it, a, b):
        return a / (it ** b)
        
#    def svm_fast_subgrad(self, w_start = None, w0 = 0, step = _const_, a = 0.01,  
#                               b = 0, max_iter= 100, verbose= False, tol= 1e-6,
#                               norma = 0):
#        self.gamma_ = 0
#        status = 1
#        M = self.X_.shape[0]
#        if w_start is None:
#            w_start = np.zeros((self.y_.shape))
#        time_ = -time.time()
#        X_ = np.hstack((self.X_, np.ones((M, 1))))
#        w_i = np.vstack((w_start, w0))
#        w_i_0 = np.vstack((w_i[:-1], 0))
#        it  = 0
#        obj_curve = []
#        rd = -1
#        random = np.random.randint(0, M, max_iter * 20)
#        while it < max_iter and status:
#            w_i_0[-1, :] = w_i[-1, :] 
#            try:            
#                it += 1
#                val = 1
#                while val >= 1:
#                    rd += 1
#                    val = (self.y_[random[rd]] *(X_[random[rd]].dot(w_i)))[0]
#                g = w_i_0 - (self.C_ * M * self.y_[random[rd]] * X_[random[rd]])[np.newaxis].T
#                w_i = w_i - g * step(it, a, b)
#                status = np.int32(dist.norm(w_i[:-1] - norma) > tol)
#                if verbose == True:
#                    self.w_ = w_i[:-1]
#                    self.b_ = w_i[-1]
#                    print(self.compute_primal_objective())
#                    obj_curve.append(self.compute_primal_objective())
#                else:
#                    norma = w_i[:-1]
#            except TypeError:
#                print("except")
#                it -= 1
#                rd = -1
#                random = np.random.randint(0, M, max_iter * 20)
#        time_ += time.time()
#        obj_curve = np.array(obj_curve)
#        return self._make_svm_result_(w = w_i[:-1], obj_curve = obj_curve,
#                                    status = status, ksi = 0, time = time_,
#                                    b = w_i[-1])
                                   
    def svm_subgradient_solver(self, w_start = None, w0 = 0, step = _base_, a = 0.01,  
                               b = 0.1, max_iter= 1000, verbose= False, tol= 1e-6,
                               rule = 'f', stochastic = False, func = 0,
                               norma = 0, sub = 0.5):
        """
        step - is a function for determinate step in each iteration
        
        rule : f(functional) or n(norma) -- rule for stopping algorithm
        if verbose == true, then obj_curve returned and used real optimum
        for 
        """
        def stoch(X, y, n):
            M = y.shape[0]
            num = np.arange(M)
            np.random.shuffle(num)
            i = round(M * n)
            elem = num[:i]
            direct = np.sum(y[elem] * X[elem], axis = 0).T
            return (M * direct / i)[np.newaxis].T
            
        def subgrad(X, y, n):
            return np.sum(y * X, axis = 0)[np.newaxis].T
        
        self.gamma_ = 0
        status = 1
        tol *= self.X_.shape[0]
        if w_start is None:
            w_start = np.zeros((self.X_.shape[1], 1))
        if stochastic == True:
            move = stoch
        else:
            move = subgrad
        if verbose == False:
            norma = np.zeros(w_start.shape)
        
        time_ = -time.time()
        X_ = np.hstack((self.X_, np.ones((self.X_.shape[0], 1))))
        w_i = np.vstack((w_start, w0))
        w_i_0 = np.vstack((w_i[:-1], 0))
        
        it  = 0
        obj_curve = []
        values = self.y_ * X_.dot(w_i)
        get_rows = np.where(values < 1)[0]
        while it < max_iter and status and len(get_rows) > 0:
            it += 1
            diract = move(X_[get_rows], self.y_[get_rows], sub)
            g = w_i_0 - self.C_ * diract    
            w_i = w_i - g * step(it, a, b)
            w_i_0 = np.vstack((w_i[:-1], 0))
            values = self.y_ * X_.dot(w_i)
            get_rows = np.where(values < 1)[0]
            if rule == 'f':
                func_i = 0.5 * w_i_0.T.dot(w_i_0)[0,0] + self.C_ * \
                         np.sum(1 - values[get_rows])
                status = np.int32(np.abs(func_i - func) > tol)   
                if verbose == True:
                    obj_curve.append(func_i)
                else:
                    func = func_i
            elif rule == 'n':
                status = np.int32(dist.norm(w_i[:-1] - norma) > tol) 
                if verbose == True:
                    func_i = 0.5 * w_i_0.T.dot(w_i_0)[0,0] + self.C_ * \
                             np.sum(1 - values[get_rows])
                    obj_curve.append(func_i)    
                else:
                    norma = w_i[-1]
            else:
                raise TypeError('no such algoritm', rule)
        time_ += time.time()
        if len(get_rows) == 0:
            status = 0
        if len(obj_curve) > 0:
            obj_curve = np.array(obj_curve)
        else:
            obj_curve = None
        return self._make_svm_result_(w = w_i[:-1], obj_curve = obj_curve,
                                    status = status, ksi = 0, time = time_,
                                    iter_ = it, b = w_i[-1])   
        
    def svm_qp_primal_solver(self, tol= 1e-6, max_iter= 100, verbose= False):
        self.gamma_ = 0
        cv.solvers.options['show_progress'] = verbose
        cv.solvers.options['maxiters'] = max_iter
        cv.solvers.options['reltol'] = tol
        cv.solvers.options['abstol'] = tol
        time_ = -time.time()
        N, D = self.X_.shape
        P = np.zeros((N + D + 1, N + D + 1))
        P[:D, :D] = np.eye(D)
        P = cv.sparse(cv.matrix(P))
        q = np.zeros((N + D + 1, 1))
        q[-N:] = self.C_
        q = cv.matrix(q)
        h = np.zeros((N + N, 1))
        h[N:] = -np.ones((N, 1))
        h = cv.matrix(h)
        G = np.zeros((2 * N, N + D + 1))
        G[:N, -N:] = - np.eye(N)
        G[N:, :] = np.hstack((- self.y_ * self.X_, -self.y_, - np.eye(N)))
        G = cv.sparse(cv.matrix(G))
        solution = cv.solvers.qp(P, q, G, h)                    
        time_ += time.time()
        answer = np.array(solution['x'])
        status = solution['iterations'] >= max_iter
        obj_curve = solution['primal objective']        
        return self._make_svm_result_(w = answer[:D], b = answer[D], 
                                    status = status, obj_curve= obj_curve,
                                    ksi = answer[D + 1:], time = time_)         
    
    def _kernel_(self, Y):
        if self.gamma_ != 0:
            ker = dist.cdist(Y, self.X_, 'sqeuclidean')
            ker *= - self.gamma_
            ker = np.exp(ker)
        else:
            ker = Y.dot(self.X_.T)        
        return ker    
    
    def svm_qp_dual_solver(self, tol=1e-6, max_iter=100, 
                           verbose=False, gamma=0):
        self.gamma_ = gamma
        cv.solvers.options['show_progress'] = verbose
        cv.solvers.options['maxiters'] = max_iter
        cv.solvers.options['reltol'] = tol
        time_ = -time.time()
        N, D = self.X_.shape
        ker = self._kernel_(self.X_)
        P = (self.y_ * ker) * self.y_.T
        P = cv.matrix(P)
        q = cv.matrix(- np.ones((N,1)))
        G = cv.sparse(cv.matrix(np.vstack((np.eye(N), -np.eye(N)))))
        h = cv.matrix(np.vstack((self.C_ * np.ones((N, 1)), np.zeros((N, 1)))))
        A = cv.matrix(self.y_.T)
        b = cv.matrix(np.zeros((1, 1)))
        solution  = cv.solvers.qp(P, q, G, h, A, b)
        P = []; G = [];
        answer = solution['x']
        A = np.array(answer)
        A[A < 1e-7] = 0
        b = np.sum(self.y_ - ker.dot(A * self.y_)) / N
        time_ += time.time()
        status = solution['iterations'] >= max_iter
        #obj_curve = solution['primal objective']        
        return self._make_svm_result_(A = A, b = b, time = time_,
                                    status = status, obj_curve= None)                                 
 
    
    def svm_liblinear_solver(self, tol= 1e-6, max_iter= 100, verbose= False):
        self.gamma_ = 0
        time_ = -time.time()
        clf = svm.LinearSVC(C = self.C_, max_iter = max_iter, tol = tol,
                            verbose = verbose)
        clf.fit(self.X_, self.y_[:,0])
        time_ += time.time()                            
        status = clf.n_iter_ == max_iter
        return self._make_svm_result_(w = clf.coef_.T, b = clf.intercept_, 
                                    time = time_, status = status)
    
    def svm_libsvm_solver(self, tol = 1e-6, max_iter = 100,
                          verbose = False, gamma = 0): 
        self.gamma_ = gamma        
        if gamma == 0:
            ker = 'linear'
        else:
            ker = 'rbf'
        time_ = -time.time()
        clf = svm.SVC(C = self.C_, kernel = ker, gamma = gamma,
                      verbose = verbose, max_iter = max_iter, tol = tol)
        clf.fit(self.X_, self.y_[:,0])
        time_ += time.time()
        A = np.zeros(self.y_.shape) 
        A[clf.support_] = np.abs(clf.dual_coef_.T)
        return self._make_svm_result_(A = A, b = clf.intercept_, time = time_)
    
    def compute_w(self):
        if type(self.A_) != np.ndarray:
            raise ValueError("compute dual problem before")
        if self.gamma_ != 0:
            raise TypeError("it is possible to count only for linear")
        self.w_ = self.X_.T.dot(self.A_ * self.y_) 
        return self.w_
    
    def compute_support_vectors(self):
        if type(self.A_) == np.ndarray:
            sup_vec = self.A_ > 1e-6
        elif type(self.w_) == np.ndarray:
            vals = self.y_ *(self.X_.dot(self.w_) + self.b_)
            if type(self.ksi_) != np.ndarray:
                self.ksi_ = np.maximum(0, 1 - vals)
            sup_vec = np.isclose(vals, 1 - self.ksi_)
            if sup_vec.any() == False:
                vals = vals / self.y_

                x_p = np.argmin(vals[np.where(vals > 0)])
                x_m = np.argmax(vals[np.where(vals < 0)])

                sup_vec = np.hstack((x_p, x_m))
        else:
            raise ValueError("compute smv before")
        self.sup_vec_ = self.X_[np.where(sup_vec)[0]]
        return self.sup_vec_
        
    def predict(self, Y):
        if self.gamma_ == 0:      
            if type(self.w_) != np.ndarray:         
                self.compute_w()       
            prediction = np.int32(Y.dot(self.w_) + self.b_ > 0)
        elif type(self.A_) == np.ndarray:
            prediction = self._kernel_(Y).dot(self.A_ * self.y_) + self.b_ > 0
            prediction = np.int32(prediction)
            prediction[prediction == 0] = -1
        return prediction
    
    def visualize(self):
        ok = True
        if type(self.X_) != np.ndarray or self.X_.shape[1] != 2:
            raise ValueError
        if type(self.sup_vec_) != np.ndarray:
#            try:
             self.compute_support_vectors()
#            except TypeError:
#                ok = False
#                warnings.warn('Can\'t show support vectors as not enough information.')
        
        h = (self.X_.max() - self.X_.min()) / 500 # step in picture       
        
        x_min, x_max = self.X_[:, 0].min() - h * 3, self.X_[:, 0].max() + h * 3
        y_min, y_max = self.X_[:, 1].min() - h * 3, self.X_[:, 1].max() + h * 3
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = self.predict(np.hstack((xx.ravel()[np.newaxis].T,
                                    yy.ravel()[np.newaxis].T)))

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        if ok == True:
            plt.scatter(self.sup_vec_[:, 0], self.sup_vec_[:, 1],
                        s=80, facecolors='none')
                    
        plt.scatter(self.X_[:, 0], self.X_[:, 1], c= self.y_,
                    cmap=plt.cm.Paired)
        # plt.show()
        return
        
    def compute_primal_objective(self):
        if type(self.w_) != np.ndarray:
            if self.gamma_ == 0 and type(self.A_) == np.ndarray:
                self.compute_w()
            else:
                raise TypeError('Before use realized method')
        val = 1 - self.y_ * (self.X_.dot(self.w_) + self.b_)
        val = np.sum(np.maximum(np.zeros((val.shape)), val))
        return 0.5 * self.w_.T.dot(self.w_)[0,0] + self.C_ * val
        
    def compute_dual_objective(self):
        if type(self.A_) != np.ndarray:
            raise TypeError('Before use realized method')
        val  = (self.A_.T * self.y_.T).dot(self._kernel_(self.X_)).\
                dot(self.A_ * self.y_)
        return np.sum(self.A_) - 0.5 * val[0,0]