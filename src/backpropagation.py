from activation_funct import *
from tensorflow.keras.utils import to_categorical

def forward(a_prev, W, b, activation_function):
    z = np.dot(W, a_prev) + b

    if activation_function == "ReLU":
        a = relu(z)
    elif activation_function == "softmax":
        a = softmax(z)
    elif activation_function == "tanh":
        a = tanh(z)

    return a, z

def forward_pass(X, param, activation_function, output_layer_activation="softmax", classes_number=0):
    store = {}
    a = X.T

    store["a0"] = a
    layers_number = len(param) // 2

    for l in range(1, layers_number + 1):
        a_prev = a
        W = param[f"W{l}"]
        b = param[f"b{l}"]

        if l < layers_number:
            a, z = forward(a_prev, W, b, activation_function)
        else:
            a, z = forward(a_prev, W, b, output_layer_activation)

        store[f"a{l}"] = a
        store[f"z{l}"] = z
        store[f"a_prev{l}"] = a_prev

    if a.shape != (classes_number, X.shape[0]):
        assert a.shape == (classes_number, X.shape[0])

    return a, store

def backward(da, cache, activation_function, lambda_reg=0, regularization=None):
    a_prev, W, z = cache
    samples_number = a_prev.shape[1]

    if activation_function == "softmax":
        dz = softmax_derivative(da, z)
    elif activation_function == "ReLU":
        dz = relu_derivative(da, z)
    elif activation_function == "tanh":
        dz = tanh_derivative(da, z)

    # We have to use samples mean on each measure because we are using mini batch
    # Compute dW = dz/dW + lambda_reg * (dOmega/dW)
    dW = (1. / samples_number) * np.dot(dz, a_prev.T)
    if regularization == "L2":
        dW += (lambda_reg / samples_number) * W     # L2 regularization gradient
    elif regularization == "L1":
        dW += (lambda_reg / samples_number) * np.sign(W)    # L1 regularization gradient

    # Compute db
    db = (1. / samples_number) * np.sum(dz, axis=1, keepdims=True)

    # Compute da_succ = W^T * grad
    grad_next_layer = np.dot(W.T, dz)

    return grad_next_layer, dW, db


def backward_pass(a, y, mem, param, activation_function="ReLU", lambda_reg=0, regularization=None):
    epsilon = 1e-16
    a = np.clip(a, epsilon, 1 - epsilon)

    layers_number = len(param) // 2
    grads = {}

    # Output layer's gradient
    classes_number = len(np.unique(y))
    y = to_categorical(y.flatten(), classes_number).T
    da = a - y

    current_cache = (mem[f"a{layers_number - 1}"], param[f"W{layers_number}"], mem[f"z{layers_number}"])
    grad_next_layer, dW, db = backward(da, current_cache, "softmax", lambda_reg=lambda_reg, regularization=regularization)

    # Store computed gradient
    grads[f"dW{layers_number}"] = dW
    grads[f"db{layers_number}"] = db

    # Backpropagation for previous layers
    for l in reversed(range(1, layers_number)):
        current_cache = (mem[f"a{l - 1}"], param[f"W{l}"], mem[f"z{l}"])
        dA_prev, dW, db = backward(grad_next_layer, current_cache, activation_function, lambda_reg=lambda_reg, regularization=regularization)

        # Store gradients for next layer in the neural network:
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

    return grads
    

def calculate_cost(a, y, param, lambda_reg, regularization):
    samples_number = len(np.unique(y))

    epsilon = 1e-16
    a = np.clip(a, epsilon, 1 - epsilon)

    classes_number = len(np.unique(y))
    y = to_categorical(np.array(y).flatten(), classes_number).T

    cross_entropy = -np.sum(np.multiply(y, np.log(a))) / samples_number

    # Adding regularization
    if regularization == "L1":
        l1_reg = 0
        for l in range(len(param) // 2):
            W = param[f"W{l + 1}"]
            l1_reg += np.sum(np.abs(W))
        l1_reg = (lambda_reg / samples_number) * l1_reg

        return cross_entropy + l1_reg
    elif regularization == "L2":
        l2_reg = 0
        for l in range(len(param) // 2):
            W = param[f"W{l + 1}"]
            l2_reg += np.sum(np.square(W))
        l2_reg = (lambda_reg / (2 * samples_number)) * l2_reg

        return cross_entropy + l2_reg
    elif regularization is None:
        return cross_entropy