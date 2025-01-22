import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameters for the model")

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=0.0005,
        help="Learning rate for the generator",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=0.001,
        help="Learning rate for the discriminator",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )

    # Model architecture hyperparameters
    parser.add_argument(
        "--n_max_nodes",
        type=int,
        default=50,
        help="Maximum number of nodes in the graph",
    )
    parser.add_argument(
        "--noise_dim", type=int, default=32, help="Dimension of the noise vector"
    )
    parser.add_argument(
        "--cond_dim",
        type=int,
        default=7,
        help="Conditioning dimension, which is 7 because we extract the 7 graph features, from the prompt as they always have the same feature order in the description.",
    )
    parser.add_argument(
        "--hidden_dim_generator",
        type=int,
        default=256,
        help="Hidden dimension for the generator",
    )
    parser.add_argument(
        "--hidden_dim2_generator",
        type=int,
        default=128,
        help="Second hidden dimension for the generator",
    )
    parser.add_argument(
        "--hidden_dim_discriminator",
        type=int,
        default=256,
        help="Hidden dimension for the discriminator",
    )
    parser.add_argument(
        "--data_aug",
        type=int,
        default=8000,
        help="Define the number of generated graph through data augmentation",
    )

    args = parser.parse_args()
    return args
