import numpy as np

def tanh(z):
    a = np.tanh(z)

    return a

def relu(z):
    a = np.maximum(0, z)

    return a

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    a = exp_z / exp_z.sum(axis=0)

    return a

def tanh_derivative(dA, z):
    a  = tanh(z)
    dZ = dA * (1 - np.square(a))

    return dZ

def relu_derivative(dA, z):
    a = relu(z)
    dZ = np.multiply(dA, np.int64(a > 0))

    return dZ

def softmax_derivative(dA, z):
    A = softmax(z)
    dZ = dA * A * (1 - A)

    return dZ