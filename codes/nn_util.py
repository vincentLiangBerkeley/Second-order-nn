import numpy as np
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