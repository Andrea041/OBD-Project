import itertools
import logging
import threading
import time

from train import train_model
from backpropagation import *

def evaluate_model(X_valid, y_valid, train_parameters, activation_function):
    class_probabilities, _ = forward_pass(X_valid, train_parameters, activation_function, "softmax", classes_number=len(np.unique(y_valid)))

    y = np.argmax(class_probabilities, axis=0)

    if len(y_valid.shape) > 1 and y_valid.shape[1] > 1:
        y_hat = np.argmax(y_valid, axis=1)
    else:
        y_hat =  np.array(y_valid).flatten()

    accuracy = np.mean(y == y_hat) * 100

    num_classes = len(np.unique(y_hat))
    precision_list, recall_list, f1_list = [], [], []

    for c in range(num_classes):
        TP = np.sum((y_hat == c) & (y == c))  # True Positives
        FP = np.sum((y_hat != c) & (y == c))  # False Positives
        FN = np.sum((y_hat == c) & (y != c))  # False Negatives

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
        accuracy, _, _, _ = evaluate_model(X_valid, y_valid, train_parameters, activation_function)

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

    from concurrent.futures import ThreadPoolExecutor

    # Supponiamo che le seguenti variabili siano già definite:
    # lambda_list, layers, train, cost_lambda_config, best_val_accuracy, best_lambda, best_parameters

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
                print(f"\nLambda: {result_lambda}, Neural Network Configuration: {nn_config}, Validation Accuracy: {val_accuracy} %, Loss: {costs}")

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
    print(f"\nTime spent in training: {total_time:.2f} seconds")

    return best_lambda, best_val_accuracy, best_parameters, cost_lambda_config, total_time