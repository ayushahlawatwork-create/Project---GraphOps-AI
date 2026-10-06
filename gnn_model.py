import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("DATASET/alibaba_microservices_large.csv")

# Create node list
nodes = pd.unique(
    pd.concat([df["source_node"], df["target_node"]])
)

node_to_id = {node: i for i, node in enumerate(nodes)}

# Convert source and target nodes to numbers
source = df["source_node"].map(node_to_id).values
target = df["target_node"].map(node_to_id).values

edge_index = torch.tensor(
    [source, target],
    dtype=torch.long
)


# =========================
# 2. NODE FEATURES
# =========================

# Maximum bottleneck value for each source node
node_features = (
    df.groupby("source_node")["bottleneck_flag"]
    .max()
    .reindex(nodes)
    .fillna(0)
    .values
)

x = torch.tensor(
    node_features,
    dtype=torch.float
).view(-1, 1)


# =========================
# 3. NODE LABELS
# =========================

y = torch.tensor(
    node_features,
    dtype=torch.long
)


# =========================
# 4. CREATE GRAPH
# =========================

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features:", data.num_node_features)
print("Labels:", data.y.shape)


# =========================
# 5. GCN MODEL
# =========================

class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)

        return x


# =========================
# 6. CREATE MODEL
# =========================

model = GCN()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

loss_function = torch.nn.CrossEntropyLoss()


# =========================
# 7. TRAINING
# =========================

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    loss = loss_function(
        output,
        data.y
    )

    loss.backward()

    optimizer.step()


# =========================
# 8. RESULT
# =========================

print("Training completed")
print("Final loss:", loss.item())
torch.save(model.state_dict(), "gnn_model.pth")