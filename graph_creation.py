import pandas as pd
import networkx as nx

alibaba = pd.read_csv("DATASET/alibaba_microservices_large.csv")

G = nx.DiGraph()

for _, row in alibaba.iterrows():
    G.add_edge(
        row["source_node"],
        row["target_node"],
        latency=row["avg_latency_ms"],
        data_transfer=row["data_transfer_mb"],
        error_rate=row["error_rate_percent"],
        bottleneck=row["bottleneck_flag"]
    )

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())