import pandas as pd

from Preprocessing import *

def print_menu(message, choice_number):
    while True:
        print(message)
        choice = input(f"Select a number between {choice_number}: ").strip().lower()

        if choice in choice_number:
            return choice
        else:
            print("Choice is not valid. Please select a valid input\n")

def main():
    # Flag for testing dataset
    feature_sel = False

    # Determining which dataset to use
    dataset_to_use = print_menu(
        "Choose the dataset you would like to use:\n" +
        "1 - Obesity\n" +
        "2 - Obesity with Feature Selection\n",
        ["1", "2"]
    )

    # Determining which activation function to use
    activation_function = print_menu(
        "Choose the activation function you would like to use:\n" +
        "1 - ReLU\n" +
        "2 - tanh\n",
        ["1", "2"]
    )

    if activation_function == "1":
        "ReLU"
    elif activation_function == "2":
        "tanh"

    # Determining which type of regularization to use
    regularization = print_menu(
        "Choose the regularization type you would like to use:\n" +
        "1 - L1\n" +
        "2 - L2\n",
        ["1", "2"]
    )

    if regularization == "1":
        "L1"
    elif regularization == "2":
        "L2"

    # Preprocess chosen dataset
    if dataset_to_use == "1" or dataset_to_use == "2":
        dataset_name = "obesity"
        dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

        # Encodings of strings in numbers
        encode_obesity_dataset(dataset)
        label = "NObeyesdad"
        test_size = 0.2
        validation_size = 0.1
        if dataset_to_use == "2":
            feature_sel = True

    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessing(dataset, label, test_size, validation_size, feature_sel)


if __name__ == "__main__":
    main()