import numpy as np
import scipy.linalg as sp

# J = [ NN[l-1] kron A[l-1]  ...  NN[k] kron A[k]  ...  NN[0] kron A[0] ]
# Ns, As are lists containing the matrices N_i and A_i
# QR_of_J finds the QR factorization of the J matrix
# TODO: pivoting
def QR_of_J(NN, A):
    currNN = list(NN)
    l = len(currNN)   
    R = [[None for j in range(l)] for i in range(l)] # list containing R factors of NN
    Q = [None for i in range(l)] # list containing Q factors of NN
    S = [[None for j in range(l)] for i in range(l)] # list containing R factors of A
    U = [None for i in range(l)] # list containing Q factors of A
    
    for i in range(len(NN)-1,-1,-1):
        n = currNN[i].shape[1]
        q,r = np.linalg.qr(currNN[i], mode='complete')
        Q[i] = q
        R[i][i] = r[:n,:]
        u,s = np.linalg.qr(A[i], mode='complete')
        U[i] = u
        S[i][i] = s
        for j in range(i-1,-1,-1):
            temp = np.dot(q.T,currNN[j])
            R[i][j] = temp[:n,:]
            currNN[j] = temp[n:,:]
            temp2 = np.dot(u.T,A[j])
            S[i][j] = temp2
    
    return [R,S]

# assemble full R matrix, for testing
def assemble_R(R, S):
    l = len(R)
    # assemble right most block column
    currCol = np.kron(R[l-1][0], S[l-1][0])
    for i in range(l-2,-1,-1):
        currCol = np.vstack((currCol, np.kron(R[i][0], S[i][0])))
    m = currCol.shape[0]
    RR = np.copy(currCol)
    # assemble other block columns, proceeding from right to left
    for j in range(1,l,1):
        currCol = np.kron(R[l-1][j], S[l-1][j])
        for i in range(l-2,j-1,-1):
            currCol = np.vstack((currCol, np.kron(R[i][j], S[i][j])))
        currCol = np.vstack((currCol, np.zeros((m-currCol.shape[0], currCol.shape[1]))))
        RR = np.hstack((currCol, RR))
    return RR

# assemble full J matrix, for testing
def assemble_J(NN, A, l):
    J = np.kron(NN[l-1], A[l-1])
    for i in range(l-2,-1,-1):        
        J = np.hstack((J, np.kron(NN[i], A[i])))
    return J

# append zero rows to blocks of S that are short fat matrices
def append_zero_rows(S):
    
    l = len(S)
    SS = [[None for j in range(l)] for i in range(l)] # make a new S, don't modify the original within the function
    
    for i in range(l):
        m,n = S[i][i].shape
        if m < n: # diagonal block is short fat, append zeros to this and all row blocks
            for j in range(i+1):
                SS[i][j] = np.vstack((S[i][j], np.zeros((n-m, S[i][j].shape[1]))))
        else:
            for j in range(i+1):
                SS[i][j] = S[i][j]
    
    return SS

# assemble preconditioner R as a full matrix, a square matrix with diagonal abs thresholding by lambda > 0
# TODO: handle case when blocks of R, S are tall skinny, for now we assume all blocks are square or short fat
def assemble_R_precon(R, S, lamb):
    
    # add appropriate zeros rows
    SS = append_zero_rows(S)
    
    # assemble R
    R_thr = assemble_R(R,SS)
    
    # apply thresholding
    for i in range(R_thr.shape[0]):
        R_thr[i][i] = np.max((R_thr[i][i], lamb)) if R_thr[i][i] >= 0 else np.min((R_thr[i][i], -lamb))
        
    return R_thr
        