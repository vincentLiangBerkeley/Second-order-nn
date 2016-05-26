import numpy as np
import scipy.linalg as sp

# threshold the abs value of the diagonal of the matrix A at lamb
# function modifies A in place
def threshold_lamb(A, lamb):
    for i in range(A.shape[0]):
        A[i,i] = np.max((A[i][i], lamb)) if A[i][i] >= 0 else np.min((A[i][i], -lamb))

# backsolve R x = b for x, without explicitly forming R
# where R,S are kron blocks of R
def backsolve_R_i(R, S, lamb, b):
    
    L = len(R)
    i_o_stop = len(b)
    x = np.zeros_like(b)
    
    # block backsolve
    for i in range(L):
        Ai = np.kron(R[i][i], S[i][i])
        threshold_lamb(Ai,lamb)
        m,n = Ai.shape
        bi = b[i_o_stop-n : i_o_stop]
        i_i_start = i_o_stop
        for j in range(i-1,-1,-1):
            Aij = np.kron(R[i][j], S[i][j])
            k,l = Aij.shape
            bi = bi - np.dot(Aij, x[i_i_start : i_i_start+l])
            i_i_start += l
        x[i_o_stop-n : i_o_stop] = sp.solve_triangular(Ai,bi)
        i_o_stop -= n
    
    return x
    
# backsolve R^T x = b for x, without explicitly forming R
# where R,S are kron blocks of R
def backsolve_Rtr_i(R, S, lamb, b):
    
    L = len(R)
    i_o_start = 0
    x = np.zeros_like(b)
    
    # block forwardsolve
    for i in range(L-1,-1,-1):
        Ai = np.kron(R[i][i].T, S[i][i].T)
        threshold_lamb(Ai, lamb)
        m,n = Ai.shape
        bi = b[i_o_start : i_o_start+n]
        i_i_stop = i_o_start
        for j in range(i+1,L,1):
            Aij = np.kron(R[j][i].T, S[j][i].T)
            k,l = Aij.shape
            bi = bi - np.dot(Aij, x[i_i_stop-l : i_i_stop])
            i_i_stop -= l
        x[i_o_start : i_o_start+n] = sp.solve_triangular(Ai,bi,lower=True)
        i_o_start += n
        
    return x
