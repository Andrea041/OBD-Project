from preprocessing import *
from costants import *
from cross_validation import *
from src.graph_utils import save_loss_plots


def print_menu(message, choice_number):
    while True:
        print(message)
        choice = input(f"Select a number between {choice_number}: ").strip().lower()

        if choice in choice_number:
            return choice
        else:
            print("Choice is not valid. Please select a valid input\n")

def main():
    # Flags for testing dataset
    feature_sel = False
    rebalancing = False

    # Determining which dataset to use
    dataset_to_use = print_menu(
        "Choose the dataset you would like to use:\n" +
        "1 - Obesity\n" +
        "2 - Obesity with Feature Selection\n" +
        "3 - Obesity with Balancing\n",
        ["1", "2", "3"]
    )

    # Determining which activation function to use
    activation_function = print_menu(
        "Choose the activation function you would like to use:\n" +
        "1 - ReLU\n" +
        "2 - tanh\n",
        ["1", "2"]
    )

    if activation_function == "1":
        activation_function = "ReLU"
    elif activation_function == "2":
        activation_function = "tanh"

    # Determining which type of regularization to use
    regularization = print_menu(
        "Choose the regularization type you would like to use:\n" +
        "0 - No regularization\n" +
        "1 - L1\n" +
        "2 - L2\n",
        ["0", "1", "2"]
    )

    if regularization == "1":
        regularization = "L1"
    elif regularization == "2":
        regularization = "L2"
    elif regularization == "0":
        regularization = None

    # Preprocess chosen dataset
    if dataset_to_use == "1" or dataset_to_use == "2" or dataset_to_use == "3":
        dataset_name = "Obesity"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        # Encodings of strings in numbers
        encode_obesity_dataset(dataset)
        label = "NObeyesdad"
        test_size = 0.2
        validation_size = 0.1
        if dataset_to_use == "2":
            feature_sel = True
        elif dataset_to_use == "3":
            rebalancing = True

    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessing(dataset, label, test_size, validation_size, feature_sel, rebalancing)

    # Choosing set of lambda for cross-validation
    if regularization == "L1":
        lambda_list = L1_LIST
    elif regularization == "L2":
        lambda_list = L2_LIST
    else:
        lambda_list = None

    # Defining layer for our neural networks
    first_layer = X_train.shape[1]
    last_layer = len(np.unique(y_train))
    neural_network_layers = [[first_layer, 32, 32, last_layer],
                             #[first_layer, 64, 32, last_layer],
                             [first_layer, 64, 64, last_layer],
                             #[first_layer, 128, 64, last_layer],
                             [first_layer, 128, 128, last_layer],
                             #[first_layer, 256, 128, last_layer],
                             [first_layer, 256, 256, last_layer],
                             #[first_layer, 512, 256, last_layer],
                             [first_layer, 512, 512, last_layer]]

    # Start cross validation
    lambda_reg, _, parameters, loss_cost, total_time = cross_validation(X_train, y_train, X_valid,
                                                                       y_valid, activation_function, lambda_list,
                                                                       neural_network_layers, regularization)

    save_loss_plots(loss_cost, dataset_name, activation_function, regularization)

    # Compute performance on test set
    accuracies = {}
    precisions = {}
    recalls = {}
    f1_nn = {}
    for nn_config in neural_network_layers:
        accuracy, precision, recall, f1 = evaluate_model(X_test, y_test, parameters[tuple(nn_config)], activation_function)

        accuracies[tuple(nn_config)] = accuracy
        precisions[tuple(nn_config)] = precision
        recalls[tuple(nn_config)] = recall
        f1_nn[tuple(nn_config)] = f1

        # Print result
        print(f"Lambda*: {lambda_reg[tuple(nn_config)]}")
        print(f"Accuracy on test set: {accuracies[tuple(nn_config)]} %")
        print(f"Precision on test set: {precisions[tuple(nn_config)]} %")
        print(f"Recall on test set: {recalls[tuple(nn_config)]} %")
        print(f"F1 Score on test set: {f1_nn[tuple(nn_config)]} %\n")

if __name__ == "__main__":
    main()