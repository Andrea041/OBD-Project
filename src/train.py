from costants import *
from parameters_inizialization import param_initializer
from batch_generator import *
from backpropagation import *

def update_parameters(parameters, grads, learning_rate, use_momentum, beta):
    layer_number = len(parameters) // 2
    parameters_prev = parameters

    for l in range(1, layer_number + 1):
        if not use_momentum:
            parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"]
            parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"]
        else:
            parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"] + beta * (
                        parameters[f"W{l}"] - parameters_prev[f"W{l}"])
            parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"] + beta * (
                        parameters[f"b{l}"] - parameters_prev[f"b{l}"])

    return parameters, parameters_prev


def train_model(X_train, y_train, layers, activation_function, lambda_reg, regularization):
    cost_per_epoch = []
    learning_rate = ALPHA
    decay_rate = DECAY_RATE

    parameters, momentum_parameters = param_initializer(activation_function, layers)

    for epoch in range(EPOCH_NUMBER):
        if DECAY_BOOL:
            learning_rate = ALPHA / (1 + decay_rate * epoch)

        mini_batches = generate_mini_batches(X_train, y_train, BATCH_SIZE)

        for X_mini_batch, y_mini_batch in mini_batches:
            a, mem = forward_pass(X_mini_batch, parameters, activation_function, classes_number=len(np.unique(y_train)))
            costs = calculate_cost(a, y_mini_batch, parameters, lambda_reg, regularization, classes_number=len(np.unique(y_train)))
            grads = backward_pass(a, y_mini_batch, mem, parameters, activation_function, lambda_reg, regularization, classes_number=len(np.unique(y_train)))
            parameters, momentum_parameters = update_parameters(parameters, grads, learning_rate, APPLY_MOMENTUM, BETA_MOMENTUM)

        # Compute y_hat
        a_epoch, epoch_mem = forward_pass(X_train, parameters, activation_function, classes_number=len(np.unique(y_train)))
        epoch_cost = calculate_cost(a_epoch, y_train, parameters, lambda_reg, regularization, classes_number=len(np.unique(y_train)))
        cost_per_epoch.append(epoch_cost)

    return parameters, cost_per_epoch