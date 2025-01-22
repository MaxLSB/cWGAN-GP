import os
import math
import networkx as nx
import numpy as np
import random
import torch
import torch.nn.functional as F
import community as community_louvain
import re
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

############################################# Useful functions #############################################


def preprocess_dataset(dataset, n_max_nodes):

    data_lst = []
    if dataset == "test":
        filename = "./data/dataset_" + dataset + ".pt"
        desc_file = "./data/" + dataset + "/test.txt"

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f"\nDataset {filename} loaded from file")

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = extract_numbers(desc)

                # For One-hot encoding the number of nodes and edges
                # vec = torch.zeros(50)
                # vec[: int(feats_stats[0])] = 1
                # edg = torch.zeros(int(50 * 50 / 2))
                # edg[: int(feats_stats[1])] = 1
                # feats_stats = vec.tolist() + edg.tolist() + feats_stats[2:]

                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(
                    Data(
                        stats=feats_stats,
                        filename=graph_id,
                        true_stats=extract_numbers(desc),
                    )
                )
            fr.close()
            torch.save(data_lst, filename)
            print(f"\nDataset {filename} saved")

    else:
        filename = "./data/dataset_" + dataset + ".pt"
        graph_path = "./data/" + dataset + "/graph"
        desc_path = "./data/" + dataset + "/description"

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f"\nDataset {filename} loaded from file")

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]

            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1 :]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")
                # load dataset to networkx
                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                        node_degree_list, key=lambda tt: tt[1], reverse=True
                    )

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
                adj = torch.from_numpy(adj_bfs).float()
                size_diff = n_max_nodes - G.number_of_nodes()
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)

                # For One-hot encoding the number of nodes and edges
                # vec = torch.zeros(50)
                # vec[: int(feats_stats[0])] = 1

                # edg = torch.zeros(int(50 * 50 / 2))
                # edg[: int(feats_stats[1])] = 1

                # feats_stats = vec.tolist() + edg.tolist() + feats_stats[2:]

                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(
                    Data(
                        A=adj,
                        stats=feats_stats,
                        filename=filen,
                    )
                )
            torch.save(data_lst, filename)
            print(f"\nDataset {filename} saved")
    return data_lst


def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r"\d+\.\d+|\d+", text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats


def generate_random_graph(max_nodes=50):
    # Choose a random number of nodes and edges
    num_nodes = random.randint(2, max_nodes)
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = random.randint(num_nodes - 1, max_possible_edges)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    random.shuffle(possible_edges)

    edges = possible_edges[:num_edges]
    graph.add_edges_from(edges)

    return graph


def augment(data_aug):
    """Function to augment the dataset by generating random graphs and computing their features"""

    filen = "graph_"
    file_num = 8001
    n_max_nodes = 50
    data_lst = []

    for _ in tqdm(range(data_aug), desc="Generating graphs"):
        G = generate_random_graph()

        if G is None:
            print("A graph is None")

        CGs = [G.subgraph(c) for c in nx.connected_components(G)]

        # rank connected componets from large to small size
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

        node_list_bfs = []
        for ii in range(len(CGs)):
            node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
            degree_sequence = sorted(
                node_degree_list, key=lambda tt: tt[1], reverse=True
            )

            bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
            node_list_bfs += list(bfs_tree.nodes())

        adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

        adj = torch.from_numpy(adj_bfs).float()
        size_diff = n_max_nodes - G.number_of_nodes()

        adj = F.pad(adj, [0, size_diff, 0, size_diff])
        adj = adj.unsqueeze(0)

        # Compute the different characteristics of the graph to create stats
        nodes = float(G.number_of_nodes())
        edges = float(G.number_of_edges())
        av_d = float(sum(dict(G.degree()).values()) / G.number_of_nodes())
        partition = community_louvain.best_partition(G)
        num_communities = float(len(set(partition.values())))
        triangles = nx.triangles(G)
        triangles = sum(triangles.values()) // 3
        global_cluster = float(nx.transitivity(G))
        core_numbers = nx.core_number(G)
        max_k_core_value = float(max(core_numbers.values()))

        feats_stats = [
            nodes,
            edges,
            av_d,
            triangles,
            global_cluster,
            max_k_core_value,
            num_communities,
        ]

        # For One-hot encoding the number of nodes and edges
        # vec = torch.zeros(50)
        # vec[: int(feats_stats[0])] = 1
        # edg = torch.zeros(int(50 * 50 / 2))
        # edg[: int(feats_stats[1])] = 1
        # feats_stats = vec.tolist() + edg.tolist() + feats_stats[2:]

        feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

        data_lst.append(
            Data(
                A=adj,
                stats=feats_stats,
                filename=filen + str(file_num),
            )
        )
        file_num += 1

    return data_lst
