import numpy as np

def param_initializer(activation_function, layers):
    if activation_function == "ReLU":
        param = initialize_parameters_he(layers)
    elif activation_function == "tanh":
        param = initialize_parameters_xavier(layers)
    else:
        print("Error: activation_function must be relu or tanh!")
        return -1, -1

    # A dictionary of zero-initialized values, useful for optimization methods like momentum.
    prev_parameters = {key: np.zeros_like(value) for key, value in param.items()}

    return param, prev_parameters

def initialize_parameters_he(layers):
    np.random.seed(3)
    parameters = {}
    layers_number = len(layers) - 1

    for l in range(1, layers_number + 1):
        parameters['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2.0 / layers[l - 1])
        parameters['b' + str(l)] = np.zeros((layers[l], 1))
        assert parameters[f"W{l}"].shape == (layers[l], layers[l - 1])
        assert parameters[f"b{l}"].shape == (layers[l], 1)

    return parameters

def initialize_parameters_xavier(layers):
    np.random.seed(3)
    parameters = {}
    layers_number = len(layers) - 1  # of layer

    for l in range(1, layers_number + 1):
        parameters[f"W{l}"] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(1 / layers[l - 1])
        parameters[f"b{l}"] = np.zeros((layers[l], 1))

        # Check shape
        assert parameters[f"W{l}"].shape == (layers[l], layers[l - 1])
        assert parameters[f"b{l}"].shape == (layers[l], 1)

    return parameters