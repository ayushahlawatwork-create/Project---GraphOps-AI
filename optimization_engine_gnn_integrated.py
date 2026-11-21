import pandas as pd
import networkx as nx
from pathlib import Path
from networkx.algorithms.community import louvain_communities


DATASET_PATHS = [
    Path("DATASET/alibaba_microservices_large.csv"),
    Path("alibaba_microservices_large.csv"),
]
PREDICTIONS_PATH = Path("gnn_all_predictions.csv")


def find_dataset():
    for path in DATASET_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find alibaba_microservices_large.csv "
        "inside DATASET/ or the project root."
    )


def load_cloud_topology():
    print("Loading network topology data...")

    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    # Keep the existing undirected graph used by the optimization engine.
    G = nx.from_pandas_edgelist(
        df,
        source="source_node",
        target="target_node",
        edge_attr=[
            "avg_latency_ms",
            "error_rate_percent",
            "bottleneck_flag",
        ],
        create_using=nx.Graph(),
    )

    # Attach GNN predictions when the prediction file is available.
    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)

        required = {
            "source_node",
            "target_node",
            "predicted_bottleneck",
            "bottleneck_probability",
        }
        missing = required - set(predictions.columns)

        if missing:
            print(
                "WARNING: GNN predictions file is missing columns:",
                sorted(missing),
            )
        else:
            # The optimizer graph is undirected, so use a canonical
            # (sorted) pair as the key for both A-B and B-A.
            # A plain dictionary avoids pandas MultiIndex lookup issues.
            prediction_values = {}

            for row in predictions.itertuples(index=False):
                key = tuple(sorted((str(row.source_node), str(row.target_node))))
                prediction_values.setdefault(key, []).append(
                    (float(row.bottleneck_probability), int(row.predicted_bottleneck))
                )

            prediction_by_edge = {}
            for key, values in prediction_values.items():
                mean_probability = sum(v[0] for v in values) / len(values)
                mean_prediction = sum(v[1] for v in values) / len(values)
                prediction_by_edge[key] = {
                    "gnn_bottleneck_probability": mean_probability,
                    "gnn_predicted_bottleneck": int(mean_prediction >= 0.5),
                }

            attached = 0

            for source, target, attributes in G.edges(data=True):
                key = tuple(sorted((str(source), str(target))))
                prediction = prediction_by_edge.get(key)

                if prediction is not None:
                    attributes["gnn_bottleneck_probability"] = float(
                        prediction["gnn_bottleneck_probability"]
                    )
                    attributes["gnn_predicted_bottleneck"] = int(
                        prediction["gnn_predicted_bottleneck"]
                    )
                    attributes["gnn_prediction_available"] = True
                    attached += 1
                else:# No test prediction exists for this edge.
                    # Preserve the original latency-based behavior.
                    attributes["gnn_bottleneck_probability"] = None
                    attributes["gnn_predicted_bottleneck"] = None
                    attributes["gnn_prediction_available"] = False

            print(
                f"Attached GNN predictions to {attached} of "
                f"{G.number_of_edges()} unique graph edges."
            )
    else:
        print(
            "WARNING: gnn_test_predictions.csv not found. "
            "Path optimization will use latency only."
        )

    print("--- GraphOps Topology Loaded ---")
    print(f"Dataset: {dataset_path}")
    print(f"Total Microservices (Nodes): {G.number_of_nodes()}")
    print(f"Total Connections (Edges): {G.number_of_edges()}")

    return G


def partition_network_clusters(G):
    print("\n--- Running Graph Clustering (Louvain) ---")

    communities = louvain_communities(
        G,
        weight="avg_latency_ms",
        seed=42,
    )

    print(
        f"Network partitioned into {len(communities)} "
        "optimized clusters."
    )

    cluster_info = {}

    for i, community in enumerate(communities):
        cluster_info[f"Cluster_{i + 1}"] = len(community)

    return cluster_info


def optimize_network_path(
    G,
    source_node,
    target_node,
    risk_penalty=2.0,
):
    print(
        f"\nFinding GNN-aware optimized path "
        f"from {source_node} to {target_node}..."
    )

    # Risk-adjusted cost increases the cost of edges that the GNN
    # considers likely to be bottlenecks. Edges without predictions
    # retain their original latency cost.
    for _, _, attributes in G.edges(data=True):
        latency = float(attributes.get("avg_latency_ms", 0.0))
        probability = attributes.get("gnn_bottleneck_probability")

        if probability is None:
            attributes["risk_adjusted_cost"] = latency
        else:
            attributes["risk_adjusted_cost"] = (
                latency * (1.0 + risk_penalty * float(probability))
            )

    try:
        best_path = nx.shortest_path(
            G,
            source=source_node,
            target=target_node,
            weight="risk_adjusted_cost",
        )

        total_cost = nx.shortest_path_length(
            G,
            source=source_node,
            target=target_node,
            weight="risk_adjusted_cost",
        )

        original_latency = sum(
            G[u][v].get("avg_latency_ms", 0.0)
            for u, v in zip(best_path[:-1], best_path[1:])
        )

        print(f"Optimized Path: {' -> '.join(best_path)}")
        print("\nGNN prediction details for selected path:")

        for u, v in zip(best_path[:-1], best_path[1:]):
            edge = G[u][v]

            print(
                f"{u} -> {v} | "
                f"Prediction available: {edge.get('gnn_prediction_available')} | "
                f"Bottleneck probability: {edge.get('gnn_bottleneck_probability')}"
            )

        print(f"Original path latency: {original_latency:.3f}")
        print(f"GNN-adjusted path cost: {total_cost:.3f}")

        return best_path, total_cost    

    except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
      print(f"Could not find path: {error}")
      return None, None


if __name__ == "__main__":
    graph = load_cloud_topology()
    clusters = partition_network_clusters(graph)

    optimize_network_path(
        graph,
        source_node="service_1",
        target_node="service_50",
    )
