import itertools
import logging
import threading
import time

from train import train_model
from backpropagation import *

import numpy as np

def evaluate_model(X_valid, y_valid, train_parameters, activation_function, classes_number=0):
    class_probabilities, _ = forward_pass(X_valid, train_parameters, activation_function, "softmax", classes_number)

    y_pred = np.argmax(class_probabilities, axis=0)
    y_true = np.ravel(y_valid)

    accuracy = np.mean(y_pred == y_true) * 100

    precision_list, recall_list, f1_list = [], [], []

    for c in range(classes_number):
        TP = np.sum((y_pred == c) & (y_true == c))  # True Positives
        FP = np.sum((y_pred == c) & (y_true != c))  # False Positives
        FN = np.sum((y_pred != c) & (y_true == c))  # False Negatives

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = np.mean(precision_list) * 100
    recall = np.mean(recall_list) * 100
    f1 = np.mean(f1_list) * 100

    return accuracy, precision, recall, f1


def cross_validation(X_train, y_train, X_valid, y_valid, activation_function, lambda_list, layers, regularization):
    cost_lambda_config = {}

    # 5 params for each array
    best_val_accuracy = {}
    best_parameters = {}
    best_lambda = {}

    stop_feedback = False

    # Function to train the model
    def train(lambda_reg, nn_config):
        train_parameters, cost = train_model(X_train, y_train, nn_config, activation_function, lambda_reg, regularization)
        accuracy, _, _, _ = evaluate_model(X_valid, y_valid, train_parameters, activation_function, classes_number=len(np.unique(y_train)))

        return lambda_reg, accuracy, train_parameters, cost

    def feedback_message():
        for frame in itertools.cycle(['Training   ', 'Training.  ', 'Training.. ', 'Training...']):
            if stop_feedback:
                break
            print(f"\r{frame}", end='', flush=True)
            time.sleep(0.5)

    # Start the timer:
    start_time = time.time()

    # Thread for terminal view
    feedback_thread = threading.Thread(target=feedback_message)
    feedback_thread.start()

    if not lambda_list:
        lambda_list = [-1] # default value to apply no regularization

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        futures = {}
        for lambda_reg in lambda_list:
            for nn_config in layers:
                future = executor.submit(train, lambda_reg, nn_config)
                futures[future] = (lambda_reg, nn_config)

        for future in futures:
            lambda_reg, nn_config = futures[future]
            try:
                result_lambda, val_accuracy, parameters, costs = future.result()
                print(f"\nLambda: {result_lambda}, Neural Network Configuration: {nn_config}, Validation Accuracy: {val_accuracy} %")

                if result_lambda not in cost_lambda_config:
                    cost_lambda_config[result_lambda] = {}
                cost_lambda_config[result_lambda][tuple(nn_config)] = costs

                if val_accuracy > best_val_accuracy.get(tuple(nn_config), -float('inf')):
                    best_val_accuracy[tuple(nn_config)] = val_accuracy
                    best_lambda[tuple(nn_config)] = result_lambda
                    best_parameters[tuple(nn_config)] = parameters

            except Exception as e:
                logging.error(f"\nLambda calculation error for Lambda={lambda_reg}, Neural Network Configuration={nn_config}", exc_info=True)

    # Stop terminal view
    stop_feedback = True
    feedback_thread.join()

    # Stop timer:
    end_time = time.time()
    total_time = end_time - start_time

    return best_lambda, best_val_accuracy, best_parameters, cost_lambda_config, total_time