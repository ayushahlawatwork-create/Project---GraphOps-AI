
import random
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from torch_geometric.nn import GCNConv


# ==========================================
# 1. REPRODUCIBILITY AND DEVICE
# ==========================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

FEATURES = [
    "avg_latency_ms",
    "data_transfer_mb",
    "error_rate_percent"
]

TARGET = "bottleneck_flag"


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "DATASET/alibaba_microservices_large.csv"
)

required_columns = [
    "source_node",
    "target_node",
    *FEATURES,
    TARGET
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required_columns).copy()
df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("bottleneck_flag must contain only 0 and 1")

print("Dataset shape:", df.shape)
print("\nBottleneck distribution:")
print(df[TARGET].value_counts())


# ==========================================
# 3. CREATE NODE IDs
# ==========================================

nodes = pd.Index(
    pd.unique(
        pd.concat(
            [df["source_node"], df["target_node"]],
            ignore_index=True
        )
    )
)

node_to_id = {
    node: i for i, node in enumerate(nodes)
}

print("\nUnique nodes:", len(nodes))


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET]
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# ==========================================
# 5. CREATE NODE FEATURES
# ==========================================

# Use training records only to calculate
# node-level average performance metrics.

source_stats = train_df[
    ["source_node"] + FEATURES
].rename(columns={"source_node": "node"})

target_stats = train_df[
    ["target_node"] + FEATURES
].rename(columns={"target_node": "node"})

combined_stats = pd.concat(
    [source_stats, target_stats],
    ignore_index=True
)

node_stats = (
    combined_stats
    .groupby("node")[FEATURES]
    .mean()
    .reindex(nodes)
    .fillna(0)
)

node_scaler = StandardScaler()

x = torch.tensor(
    node_scaler.fit_transform(node_stats),
    dtype=torch.float
).to(device)


# ==========================================
# 6. CREATE TRAINING GRAPH
# ==========================================

train_source = (
    train_df["source_node"].map(node_to_id).values
)

train_target = (
    train_df["target_node"].map(node_to_id).values
)

# Add reverse edges for bidirectional
# message passing.

edge_index = torch.tensor(
    [
        np.concatenate([train_source, train_target]),
        np.concatenate([train_target, train_source])
    ],
    dtype=torch.long
).to(device)

train_edge_pairs = torch.tensor(
    [train_source, train_target],
    dtype=torch.long
).to(device)

test_source = (
    test_df["source_node"].map(node_to_id).values
)

test_target = (
    test_df["target_node"].map(node_to_id).values
)

test_edge_pairs = torch.tensor(
    [test_source, test_target],
    dtype=torch.long
).to(device)


# ==========================================
# 7. EDGE FEATURES AND LABELS
# ==========================================

edge_scaler = StandardScaler()

train_edge_features = edge_scaler.fit_transform(
    train_df[FEATURES]
)

test_edge_features = edge_scaler.transform(
    test_df[FEATURES]
)

train_edge_features = torch.tensor(
    train_edge_features,
    dtype=torch.float
).to(device)

test_edge_features = torch.tensor(
    test_edge_features,
    dtype=torch.float
).to(device)

y_train = torch.tensor(
    train_df[TARGET].values,
    dtype=torch.long
).to(device)

y_test = torch.tensor(
    test_df[TARGET].values,
    dtype=torch.long
).to(device)


# ==========================================
# 8. DEFINE GCN MODEL
# ==========================================

class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(35, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 2)
        )

    def forward(
        self,
        x,
        edge_index,
        edge_pairs,
        edge_features
    ):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        source_embedding = x[edge_pairs[0]]
        target_embedding = x[edge_pairs[1]]

        combined = torch.cat(
            [
                source_embedding,
                target_embedding,
                edge_features
            ],
            dim=1
        )

        return self.classifier(combined)


# ==========================================
# 9. CREATE MODEL AND OPTIMIZER
# ==========================================

model = GCN().to(device)

class_counts = np.bincount(
    train_df[TARGET].values,
    minlength=2
)

class_weights = len(train_df) / (
    2.0 * np.maximum(class_counts, 1)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

loss_function = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0005
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\nStarting GCN training...")

for epoch in range(1, 201):

    model.train()
    optimizer.zero_grad()

    output = model(
        x,
        edge_index,
        train_edge_pairs,
        train_edge_features
    )

    loss = loss_function(
        output,
        y_train
    )

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch}/200 | "
            f"Loss: {loss.item():.4f}"
        )


# ==========================================
# 11. TEST MODEL
# ==========================================

model.eval()

with torch.no_grad():

    test_output = model(
        x,
        edge_index,
        test_edge_pairs,
        test_edge_features
    )

    probabilities = torch.softmax(
        test_output,
        dim=1
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).long().cpu().numpy()

y_true = y_test.cpu().numpy()

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(
    y_true, predictions, zero_division=0
)
recall = recall_score(
    y_true, predictions, zero_division=0
)
f1 = f1_score(
    y_true, predictions, zero_division=0
)


# ==========================================
# 12. DISPLAY EVALUATION RESULTS
# ==========================================

print("\n========== GCN EVALUATION ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Bottleneck"],
        zero_division=0
    )
)

if len(np.unique(y_true)) == 2:
    auc = roc_auc_score(
        y_true,
        probabilities.cpu().numpy()
    )
    print(f"ROC-AUC:   {auc:.4f}")


# ==========================================
# 13. SAVE PREDICTIONS
# ==========================================

results = test_df.copy()

results["predicted_bottleneck"] = predictions
results["bottleneck_probability"] = (
    probabilities.cpu().numpy()
)

results.to_csv(
    "gnn_test_predictions.csv",
    index=False
)


# ==========================================
# 14. SAVE MODEL AND PREPROCESSING
# ==========================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_features": FEATURES,
        "hidden_features": 16,
        "output_classes": 2
    },
    "gnn_edge_model.pth"
)

joblib.dump(
    {
        "node_scaler": node_scaler,
        "edge_scaler": edge_scaler,
        "node_to_id": node_to_id,
        "features": FEATURES
    },
    "gnn_edge_preprocessing.pkl"
)

print("\nTraining and evaluation completed!")
print("Saved: gnn_edge_model.pth")
print("Saved: gnn_edge_preprocessing.pkl")
print("Saved: gnn_test_predictions.csv")
# ==========================================
# 15. PREDICT ALL DATASET CONNECTIONS
# ==========================================

model.eval()

all_source = df["source_node"].map(node_to_id).values
all_target = df["target_node"].map(node_to_id).values

all_edge_pairs = torch.tensor(
    [all_source, all_target],
    dtype=torch.long
).to(device)

all_edge_features = torch.tensor(
    edge_scaler.transform(df[FEATURES]),
    dtype=torch.float
).to(device)

with torch.no_grad():
    all_output = model(
        x,
        edge_index,
        all_edge_pairs,
        all_edge_features
    )

    all_probabilities = torch.softmax(
        all_output, dim=1
    )[:, 1]

    all_predictions = (
        all_probabilities >= 0.5
    ).long().cpu().numpy()

all_results = df.copy()

all_results["predicted_bottleneck"] = all_predictions
all_results["bottleneck_probability"] = (
    all_probabilities.cpu().numpy()
)

all_results.to_csv(
    "gnn_all_predictions.csv",
    index=False
)

print("Saved: gnn_all_predictions.csv")
print("Total connections predicted:", len(all_results))
# ==========================================
# 16. EXPORT GRAPH NODE EMBEDDINGS
# ==========================================

model.eval()

with torch.no_grad():
    embeddings = model.conv1(x, edge_index)
    embeddings = torch.relu(embeddings)

    embeddings = model.conv2(embeddings, edge_index)
    embeddings = torch.relu(embeddings)

embedding_df = pd.DataFrame(
    embeddings.cpu().numpy(),
    index=nodes
)

embedding_df.index.name = "service_node"

embedding_df.to_csv("gnn_node_embeddings.csv")

print("Saved: gnn_node_embeddings.csv")
print("Total node embeddings:", len(embedding_df))
print("Embedding dimensions:", embedding_df.shape[1])