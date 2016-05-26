import numpy as np
import scipy.linalg as sp

# computes (J^T J + lambda I)x
def multiply(J, lamb, x):
    return np.dot(J.T, np.dot(J,x)) + lamb*x
    
# solves (R^T R)x = b for x, where R is upper triangular
def backsolve(R, b):
    #return np.linalg.solve(R, np.linalg.solve(R.T, b))
    return sp.solve_triangular(R, sp.solve_triangular(R, b, trans=1))

# note that multiply_J_i and multiply_Jtr_i will be obsolete after implementation of nn since this will 
# become back propagation

# multiply y = J x without explicitly forming J, the implicit version of multiply_J
# J = [ NN[l-1] kron A[l-1]  ...  NN[k] kron A[k]  ...  NN[0] kron A[0] ]
# returns y = J x
def multiply_J_i(NN, A, x):
    
    l = len(NN)
    i_start = 0
    M = np.kron(NN[l-1], A[l-1])
    m,n = M.shape
    y = np.dot(M, x[i_start : i_start+n])
    i_start += n
    for i in range(l-2,-1,-1):
        M = np.kron(NN[i], A[i])
        m,n = M.shape
        y = y + np.dot(M, x[i_start : i_start+n])
        i_start += n
    return y

# returns y = J^T x
def multiply_Jtr_i(NN, A, x):
    
    l = len(NN)
    y = np.dot(np.kron(NN[l-1].T, A[l-1].T), x)
    for i in range(l-2,-1,-1):
        y = np.vstack((y, np.dot(np.kron(NN[i].T, A[i].T), x)))
    return y
    
# simple implementation of cg with preconditioning, specific to the case of solving
# A = J^T J + lambda I
# with preconditioner R such that R^T R approximates A
def pcg(J, lamb, R, b, x0, tol = 1.0e-10, max_it = 200):

    # initializations    
    x = x0
    r0 = b - multiply(J, lamb, x)
    rtilde0 = backsolve(R,r0)
    p = rtilde0

    # iterations    
    for i in range(max_it):
        # There is potential inefficiency here
        a = float(np.dot(r0.T, rtilde0)/np.dot(p.T, multiply(J, lamb, p)))
        x = x + p*a
        r = r0 - a*multiply(J, lamb, p)

        if np.linalg.norm(r) < tol:
            return x, i
        
        rtilde = backsolve(R,r)
        b = float(np.dot(r.T, rtilde)/np.dot(r0.T, rtilde0))
        p = rtilde + b * p
        r0 = r
        rtilde0 = rtilde
    return x, i