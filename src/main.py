import ipaddress

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
        "3 - Obesity with Balancing\n" +
        "4 - Predictive maintenance\n" +
        "5 - Customer segmentation\n" +
        "6 - Air quality and pollution assessment\n" +
        "7 - Sloan Digital Sky Survey - DR18\n" +
        "8 - Dry bean\n" +
        "9 - Android Malware detection\n",
        ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
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
    if dataset_to_use == "1" or dataset_to_use == "2" or dataset_to_use == "3": # Tenere
        dataset_name = "Obesity"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        # Encodings of strings in numbers
        label = "NObeyesdad"
        dataset, _ = encode_dataset(dataset, label)
        if dataset_to_use == "2":
            feature_sel = True
        elif dataset_to_use == "3":
            rebalancing = True
    elif dataset_to_use == "4":
        dataset_name = "predictive_maintenance"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        label = "Failure Type"
        dataset, _ = encode_dataset(dataset, label)
    elif dataset_to_use == "5":
        dataset_name = "customer_segmentation"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        label = "Var_1"
        dataset, _ = encode_dataset(dataset, label)
    elif dataset_to_use == "6":
        dataset_name = "updated_pollution_dataset"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        label = "Air Quality"
        dataset, _ = encode_dataset(dataset, label)
    elif dataset_to_use == "7": # Tenere
        dataset_name = "SDSS_DR18"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        label = "class"
        dataset, _ = encode_dataset(dataset, label)
    elif dataset_to_use == "8":
        dataset_name = "Dry_Bean_Dataset"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")
        label = "Class"
        dataset, _ = encode_dataset(dataset, label)
    elif dataset_to_use == "9":
        def ip_to_int(ip):
            try:
                ip = ipaddress.ip_address(ip)
                return int(ip)
            except ValueError:
                print(f"\nInvalid ip addresses found: {ip}")
                return np.nan

        dataset_name = "Android_Malware"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv", low_memory=False)
        label = "Label"
        dataset.drop(['Unnamed: 0','Flow ID',' Timestamp',' CWE Flag Count',' Down/Up Ratio','Fwd Avg Bytes/Bulk'], axis=1, inplace=True)
        dataset[' Source IP'] = dataset[' Source IP'].apply(ip_to_int)
        dataset[' Destination IP'] = dataset[' Destination IP'].apply(ip_to_int)
        dataset, _ = encode_dataset(dataset, label)

    test_size = 0.2
    validation_size = 0.1
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
                             #[first_layer, 32, 64, last_layer],
                             [first_layer, 64, 64, last_layer],
                             #[first_layer, 64, 128, last_layer],
                             [first_layer, 128, 128, last_layer],
                             #[first_layer, 128, 256, last_layer],
                             [first_layer, 256, 256, last_layer],
                             #[first_layer, 256, 512, last_layer],
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
        accuracy, precision, recall, f1 = evaluate_model(X_test, y_test, parameters[tuple(nn_config)], activation_function, classes_number=len(np.unique(y_train)))

        accuracies[tuple(nn_config)] = accuracy
        precisions[tuple(nn_config)] = precision
        recalls[tuple(nn_config)] = recall
        f1_nn[tuple(nn_config)] = f1

        # Print result
        print(f"Lambda: {lambda_reg[tuple(nn_config)]}")
        print(f"Optimal neural network configuration: {nn_config}")
        print(f"Accuracy on test set: {accuracies[tuple(nn_config)]} %")
        print(f"Precision on test set: {precisions[tuple(nn_config)]} %")
        print(f"Recall on test set: {recalls[tuple(nn_config)]} %")
        print(f"F1 Score on test set: {f1_nn[tuple(nn_config)]} %\n")

if __name__ == "__main__":
    main()