import numpy as np
from factorize import *
# Now define the most primitive functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_gradient(x):
    return 1 - tanh(x) * tanh(x)

def relu(x):
    return [i if i > 0 else 0 for i in x]

def relu_gradient(x):
    return [1 if i > 0 else 0 for i in x]

def id(x):
    return x;

def id_gradient(x):
    return 1;

# Build a toy network without bias terms
# Returns Ns as a list of matrices
# W is a list of matrices
# N0 = input data matrix
# F = list of activation functions
def forward_params(W, N0, F):
    N = [N0]
    l = len(W)
    for i in range(l):
        N.append(F[i](np.dot(W[i], N[i])))
    return N

# for ease of testing
# len(dk) = 3
def construct_2layer_nn(dk, F_func=[sigmoid, tanh], F_grad=[sigmoid_gradient, tanh_gradient]):
    ''' example usage: 
    dk = np.array([40, 20, 10])
    N, W, F, NN_id, A_id, R_id, SS_id, NN_rk1, A_rk1, R_rk1, SS_rk1, x0, b = construct_2layer_nn(dk)
    '''
    l = 3
    dx = np.sum(dk)
    
    # generate weights
    N_0 = np.random.randn(dk[0], dx)
    W_1 = np.random.randn(dk[1], dk[0])
    W_2 = np.random.randn(dk[2], dk[1])
    W = [W_1, W_2]

    # construct random right hand side
    m = W_1.shape[0]*W_1.shape[1] + W_2.shape[0]*W_2.shape[1]
    b = np.random.randn(m,1)
    x0 = np.zeros((m,1))
    
    
    # generate J, assuming identity activation function
    N = forward_params(W, N_0, F_func)
    NN_id = [n.T for n in N][:-1]
    A_id = [W_2, np.eye(dk[l-1])]
    
    R_id,S_id = QR_of_J(NN_id,A_id)
    SS_id = append_zero_rows(S_id)
    
    # generating J, assuming rank-1 approximation to activation function
    F1 = F_grad[0](np.dot(W_1, N[0]))
    u1,S1,v1 = np.linalg.svd(F1)
    g1 = np.multiply(S1[0],u1[:,0])
    h1 = v1[0,:]

    F2 = F_grad[1](np.dot(W_2, N[1]))
    u2,S2,v2 = np.linalg.svd(F2)
    g2 = np.multiply(S2[0], u2[:,0])
    h2 = v2[0,:]
    
    NN_0 = np.dot(np.diag(np.multiply(h1,h2)), N[0].T)
    NN_1 = np.dot(np.diag(h2), N[1].T)
    A_0 = np.dot(np.diag(g2), np.dot(W_2, np.diag(g1)))
    A_1 = np.diag(g2)
    NN_rk1 = [NN_0, NN_1]
    A_rk1 = [A_0, A_1]

    R_rk1,S_rk1 = QR_of_J(NN_rk1,A_rk1)
    SS_rk1 = append_zero_rows(S_rk1)
    
    return N, W, F_grad, NN_id, A_id, R_id, SS_id, NN_rk1, A_rk1, R_rk1, SS_rk1, x0, b
