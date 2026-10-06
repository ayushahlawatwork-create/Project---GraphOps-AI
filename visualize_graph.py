import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load Alibaba dataset
df = pd.read_csv("DATASET/alibaba_microservices_large.csv")

# Keep only bottleneck connections
bottleneck_df = df[df["bottleneck_flag"] == 1]

# Create graph
G = nx.DiGraph()

for _, row in bottleneck_df.iterrows():
    G.add_edge(
        row["source_node"],
        row["target_node"]
    )

print("Bottleneck nodes:", G.number_of_nodes())
print("Bottleneck edges:", G.number_of_edges())

# Layout
pos = nx.spring_layout(
    G,
    seed=42,
    k=0.8
)

# Create figure
plt.figure(figsize=(14, 10))

# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=700
)

# Draw bottleneck edges
nx.draw_networkx_edges(
    G,
    pos,
    width=2,
    arrows=True,
    arrowsize=12,
    alpha=0.8
)

# Draw labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=8
)

plt.title(
    "Alibaba Microservices Bottleneck Network",
    fontsize=18
)

plt.axis("off")

# Save graph
plt.savefig(
    "alibaba_bottleneck_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()