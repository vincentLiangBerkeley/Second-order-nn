import numpy as np
from tri_solve import *
# returns Y = J times X, Y = list of matrices
# N = list of matrices for layers
# W = list of matrices for weights
# F = list of functions, derivative of activation function
# X = list of matrices for the vector to be multiplied
def multiply_J(N, W, F, X):
    l = len(W)
    Y_old = np.zeros(N[0].shape)
    for i in range(l):
        # print(W[i].shape, N[i].shape, X[i].shape, Y_old.shape)
        Y_new = np.multiply(F[i](np.dot(W[i],N[i])), (np.dot(X[i], N[i]) + np.dot(W[i], Y_old)))
        Y_old = Y_new
    return Y_new

# returns X = list of matrices [X_l, .... X_0]
def multiply_Jtr(N, W, F, X):
    l = len(W)
    Z_old = np.multiply(F[-1](np.dot(W[l-1], N[l-1])), X)
    Xlist = [np.dot(Z_old, N[l-1].T)]
    for i in range(l-2,-1,-1):
        Z_new = np.multiply(F[i](np.dot(W[i], N[i])), (np.dot(W[i+1].T, Z_old)))
        Xlist.append(np.dot(Z_new, N[i].T))
        Z_old = Z_new
    Xlist.reverse()
    return Xlist

# function for testing, will form the Jtr matrix explicitly
def form_Jtr(N, W, F):
    n = []
    J_t = []
    for w in W:
        n.append(w.shape[0]*w.shape[1])
    total = sum(n)
    E = np.zeros((total, 1))
    for i in range(total):
        E[i] = 1
        X = []
        start = 0
        for j in range(len(W)):
            X.append(np.reshape(E[start:start + n[j]], W[j].shape, order='F'))
            start += n[j]
        J_t.append(multiply_J(N, W, F, X).flatten('F'))
        E[i] = 0
    return np.matrix(J_t)

def mat_to_vec(Xlist):
    vec = np.matrix(Xlist[0].flatten('F')).T
    for i in range(1,len(Xlist)):
        vec = np.vstack((vec, np.matrix(Xlist[i].flatten('F')).T))
    return vec

def vec_to_mat(vec, shapes):
    Xlist = []
    start = 0
    for shape in shapes:
        n = shape[0]*shape[1]
        Xlist.append(np.reshape(vec[start:start + n], shape, order='F'))
        start += n
    return Xlist

    
# threshold the abs value of the diagonal of the matrix A at lamb
# function modifies A in place
def threshold_lamb(A, lamb):
    for i in range(A.shape[0]):
        A[i,i] = np.max((A[i][i], lamb)) if A[i][i] >= 0 else np.min((A[i][i], -lamb))

# computes (J^T J + lambda I)x
def multiply_A_i(N, W, F, lamb, x):
    shapes = [w.shape for w in W]
    Xlist = vec_to_mat(x, shapes)
    # print('--',len(Xlist), Xlist[0].shape, Xlist[1].shape)
    Y = multiply_Jtr(N, W, F, multiply_J(N, W, F, Xlist))
    y = mat_to_vec(Y)
    
    return y + lamb*x

    
# solves (R^T R)x = b for x, where R is upper triangular
def backsolve_i(R, S, lamb, b):
    return backsolve_R_i(R, S, lamb, backsolve_Rtr_i(R, S, lamb, b))

# simple implementation of cg with preconditioning, specific to the case of solving
# A = J^T J + lambda I
# with preconditioner R such that R^T R approximates A
def pcg_i(N, W, F, lamb, R, S, b, x0, tol = 1.0e-1, max_it = 500, is_p = True):    
    # initializations    
    x = x0
    #r0 = b - multiply_A_i(N, W, F, lamb, x)
    r0 = b - multiply_A_i(N, W, F, lamb, x)
    rtilde0 = backsolve_i(R, S, np.sqrt(lamb),r0) if is_p else r0
    p = rtilde0

    # iterations    
    for i in range(max_it):
        v = multiply_A_i(N, W, F, lamb, p)
        a = float(np.dot(r0.T, rtilde0)/np.dot(p.T, v))
        x = x + p*a
        r = r0 - a*v

        if np.linalg.norm(r) < tol:
            return x, i
        
        rtilde = backsolve_i(R, S, np.sqrt(lamb), r) if is_p else r
        b = float(np.dot(r.T, rtilde)/np.dot(r0.T, rtilde0))
        p = rtilde + b * p
        r0 = r
        rtilde0 = rtilde
    return x, i
        