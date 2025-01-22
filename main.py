import numpy as np
import torch
import random
from tqdm import tqdm
from datetime import datetime
import networkx as nx
from torch_geometric.loader import DataLoader

from utils import preprocess_dataset, augment
from models import Generator, Discriminator, gradient_penalty
from utils import construct_nx_from_adj, preprocess_dataset
from eval_mae import eval_mae, g_to_stats
from config import parse_arguments


def main():

    #########################################    Loading Datasets    #########################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    trainset = preprocess_dataset("train", args.n_max_nodes)
    validset = preprocess_dataset("valid", args.n_max_nodes)
    testset = preprocess_dataset("test", args.n_max_nodes)

    print("\nThe datasets have been loaded !")

    trainset = trainset + augment(args.data_aug)  # Augment the training set
    random.shuffle(trainset)

    print(f"\nData Augmentation is done! Trainset size: {len(trainset)}.")

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    #########################################    Initializing the models    #########################################

    generator = Generator(
        noise_dim=args.noise_dim,
        cond_dim=args.cond_dim,
        output_dim=args.n_max_nodes,
        hidden_dim=args.hidden_dim_generator,
        hidden_dim2=args.hidden_dim2_generator,
    ).to(device)

    discriminator = Discriminator(
        input_dim=args.n_max_nodes * args.n_max_nodes,
        cond_dim=args.cond_dim,
        hidden_dim=args.hidden_dim_discriminator,
    ).to(device)

    #########################################    Training Loop    #########################################

    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr
    )
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.generator_lr)

    for epoch in range(args.num_epochs):
        train_loss_d = 0.0
        train_loss_g = 0.0
        cnt_train = 0

        discriminator.train()
        generator.train()

        for data in train_loader:
            data = data.to(device)
            real_samples = data.A
            conditions = data.stats
            batch_size = real_samples.size(0)
            cnt_train += 1

            # Train Discriminator
            discriminator.zero_grad()
            noise = torch.randn((batch_size, args.noise_dim), device=device)
            generated_samples = generator(noise, conditions).detach()

            real_output = discriminator(real_samples, conditions)
            fake_output = discriminator(generated_samples, conditions)

            gp = gradient_penalty(
                discriminator, real_samples, generated_samples, conditions, device
            )
            loss_discriminator = fake_output.mean() - real_output.mean() + 10 * gp

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Train Generator
            generator.zero_grad()
            noise = torch.randn((batch_size, args.noise_dim), device=device)
            generated_samples = generator(noise, conditions)
            fake_output = discriminator(generated_samples, conditions)

            loss_generator = -fake_output.mean()
            loss_generator.backward()
            optimizer_generator.step()

            train_loss_d += loss_discriminator.item()
            train_loss_g += loss_generator.item()

        train_loss_d /= cnt_train
        train_loss_g /= cnt_train

        discriminator.eval()
        generator.eval()
        val_loss_d = 0.0
        val_loss_g = 0.0
        cnt_val = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                real_samples = data.A
                conditions = data.stats
                batch_size = real_samples.size(0)
                cnt_val += 1

                noise = torch.randn((batch_size, args.noise_dim), device=device)
                generated_samples = generator(noise, conditions)
                real_output = discriminator(real_samples, conditions)
                fake_output = discriminator(generated_samples, conditions)

                loss_discriminator = fake_output.mean() - real_output.mean() + 10 * gp
                val_loss_d += loss_discriminator.item()

                fake_output = discriminator(generated_samples, conditions)
                loss_generator = -fake_output.mean()
                val_loss_g += loss_generator.item()

        val_loss_d /= cnt_val
        val_loss_g /= cnt_val

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(
                "{} Epoch: {:04d}, Train Loss Discriminator: {:.4f}, Train Loss Generator: {:.4f}, Val Loss Discriminator: {:.4f}, Val Loss Generator: {:.4f}".format(
                    dt_t,
                    epoch + 1,
                    train_loss_d,
                    train_loss_g,
                    val_loss_d,
                    val_loss_g,
                )
            )

    #########################################    Testing Loop    #########################################

    # Evaluating its performance with the MAE metric
    del train_loader, val_loader
    output_stats = []
    label_stats = []

    for _, data in enumerate(tqdm(test_loader, desc="Processing test set")):
        data = data.to(device)

        conditions = data.stats
        bs = conditions.size(0)

        # Generate samples using your generator
        noise = torch.randn((bs, args.noise_dim), device=device)
        adj = generator(noise, conditions)
        stat_d = torch.reshape(conditions, (-1, args.cond_dim))

        true_stats = torch.reshape(
            torch.tensor(data.true_stats), (-1, 7)
        )  # Reshape true_stats

        for i in range(bs):
            # Extract the adjacency matrix for the current graph
            adj_matrix = adj[i, :, :].detach().cpu().numpy()

            # Construct a NetworkX graph from the adjacency matrix
            Gs_generated = construct_nx_from_adj(adj_matrix)
            Gs_generated = nx.convert_node_labels_to_integers(
                Gs_generated, ordering="sorted"
            )

            # Append stats to output_stats and label_stats
            output_stats.append(g_to_stats(Gs_generated))  # Save generated graph stats
            label_stats.append(true_stats[i])  # Save true stats

    # Display the final mean absolute error
    label_stats = torch.stack(label_stats)
    output_stats = np.array(output_stats, dtype=np.float32)
    label_stats = label_stats.cpu().numpy()
    print(f"Final MAE: {eval_mae(output_stats, label_stats):2f}")


if __name__ == "__main__":
    main()
