import networkx as nx
import numpy as np
import community as community_louvain
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

############################################# MAE Benchmark Evaluation #############################################


def g_to_stats(G):
    """Function to convert a NetworkX graph to a list of graph features"""

    # Compute graph features
    nodes = float(G.number_of_nodes())
    edges = float(G.number_of_edges())
    av_d = float(sum(dict(G.degree()).values()) / G.number_of_nodes())

    # Compute community structure using Louvain method
    partition = community_louvain.best_partition(G)
    num_communities = float(len(set(partition.values())))

    # Compute number of triangles
    triangles = nx.triangles(G)
    triangles = float(sum(triangles.values()) // 3)  # Convert to float for consistency

    # Compute global clustering coefficient
    global_cluster = float(nx.transitivity(G))

    # Compute maximum k-core value
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
    return feats_stats


def eval_mae(output_stats, stat_x):
    """Function to evaluate the mean absolute error between the predicted and true graph features"""

    # Normalize each feature independently
    scaler = MinMaxScaler()
    output_stats_normalized = scaler.fit_transform(output_stats)
    stat_x_normalized = scaler.fit_transform(stat_x)
    mae_values = [
        mean_absolute_error(true_stats, feats_stats)
        for true_stats, feats_stats in zip(stat_x_normalized, output_stats_normalized)
    ]
    avg_mae = np.mean(mae_values)
    return avg_mae
