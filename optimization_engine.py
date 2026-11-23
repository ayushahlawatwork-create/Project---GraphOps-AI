import json
import os
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

def load_ml_predictions():
    """Loads Yatharth's GNN edge predictions & Akshat's LSTM traffic forecasts."""
    gnn_df = pd.read_csv("gnn_all_predictions.csv") if os.path.exists("gnn_all_predictions.csv") else None
    
    lstm_bursts = []
    if os.path.exists("outputs/latest_forecasts.json"):
        with open("outputs/latest_forecasts.json", "r") as f:
            forecasts = json.load(f)
            lstm_bursts = [
                item for item in forecasts
                if item.get("burst_status") not in (None, "normal", "LOW")
            ]
    return gnn_df, lstm_bursts

def load_cloud_topology():
    print("Loading network topology + Late-Fusion ML predictions...")
    gnn_df, lstm_bursts = load_ml_predictions()
    
    # Use Yatharth's GNN-enriched topology if present, else fallback to raw topology
    df = gnn_df.copy() if gnn_df is not None else pd.read_csv("alibaba_microservices_large.csv")
    if "predicted_bottleneck" not in df.columns:
        df["predicted_bottleneck"] = df.get("bottleneck_flag", 0)
    if "bottleneck_probability" not in df.columns:
        df["bottleneck_probability"] = 0.0

    # Late-Fusion Edge Weighting:
    # Penalize edge latency when GNN predicts a bottleneck or LSTM detects active traffic bursts
    burst_multiplier = 1.25 if len(lstm_bursts) > 0 else 1.0
    df["effective_latency_ms"] = (
        df["avg_latency_ms"]
        * (1.0 + (df["predicted_bottleneck"] * 2.5) + (df["bottleneck_probability"] * 1.5))
        * burst_multiplier
    ).round(2)

    G = nx.from_pandas_edgelist(
        df,
        source="source_node",
        target="target_node",
        edge_attr=[
            "avg_latency_ms",
            "effective_latency_ms",
            "error_rate_percent",
            "bottleneck_flag",
            "predicted_bottleneck",
            "bottleneck_probability",
        ],
        create_using=nx.Graph(),
    )
    G.graph["lstm_active_bursts"] = len(lstm_bursts)
    G.graph["gnn_flagged_edges"] = int(df["predicted_bottleneck"].sum())
    print(f"--- Late-Fusion Topology Loaded: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges ---")
    print(f"--- GNN Bottlenecks Flagged: {G.graph['gnn_flagged_edges']} | LSTM Burst Alerts: {G.graph['lstm_active_bursts']} ---")
    return G

def partition_network_clusters(G):
    communities = louvain_communities(G, weight="effective_latency_ms")
    return {f"Cluster_{i+1}": len(c) for i, c in enumerate(communities)}

def optimize_network_path(G, source_node, target_node):
    try:
        # Baseline path (raw latency only, ignoring ML predictions)
        raw_path = nx.shortest_path(G, source=source_node, target=target_node, weight="avg_latency_ms")
        raw_latency = nx.shortest_path_length(G, source=source_node, target=target_node, weight="avg_latency_ms")

        # Late-Fusion AI path (avoids GNN bottlenecks & LSTM traffic bursts)
        fused_path = nx.shortest_path(G, source=source_node, target=target_node, weight="effective_latency_ms")
        fused_latency = nx.shortest_path_length(G, source=source_node, target=target_node, weight="effective_latency_ms")
        return {
            "optimized_path": fused_path,
            "baseline_path": raw_path,
            "total_estimated_latency_ms": round(fused_latency, 2),
            "baseline_latency_ms": round(raw_latency, 2),
            "rerouted_around_anomaly": fused_path != raw_path,
        }
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None