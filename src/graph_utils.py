import matplotlib.pyplot as plt
import os

def save_loss_plots(lossCost, dataset_name, activation_function, regularization_type):
    # Create the output folder if it does not exist
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the folder for the dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create folder for activation function (ReLU or Tanh)
    activation_dir = os.path.join(dataset_dir, activation_function)
    if not os.path.exists(activation_dir):
        os.makedirs(activation_dir)

    # Create folder for regularization type (L1 or L2)
    regularization_dir = os.path.join(activation_dir, regularization_type)
    if not os.path.exists(regularization_dir):
        os.makedirs(regularization_dir)

    # Save graphs for each lambda and nn_config
    for lambd, nn_configs in lossCost.items():
        # Create the subfolder for the lambda value
        lambd_dir = os.path.join(regularization_dir, f"lambda_{lambd}")
        if not os.path.exists(lambd_dir):
            os.makedirs(lambd_dir)

        for nn_config, loss in nn_configs.items():
            # Crea loss graphs
            plt.plot(loss)
            plt.title(f"Loss for lambda = {lambd}, NN Config = {nn_config}")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            # Converti la configurazione in una stringa per il nome file
            nn_config_str = "_".join(map(str, nn_config))
            plot_filename = os.path.join(lambd_dir, f"loss_lambda_{lambd}_nn_{nn_config_str}.png")

            # Save graph
            plt.savefig(plot_filename)
            plt.close()

    print(f"Graphs saved in {regularization_dir}")