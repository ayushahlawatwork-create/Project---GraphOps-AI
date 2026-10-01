import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

def load_cloud_topology():
    print("Loading network topology data...")
    df = pd.read_csv("alibaba_microservices_large.csv")
    
    # Graph create kar rahe hain (Undirected for clustering)
    G = nx.from_pandas_edgelist(
        df, 
        source='source_node', 
        target='target_node', 
        edge_attr=['avg_latency_ms', 'error_rate_percent', 'bottleneck_flag'], 
        create_using=nx.Graph()
    )
    
    print("--- GraphOps Topology Loaded ---")
    print(f"Total Microservices (Nodes): {G.number_of_nodes()}")
    return G

def partition_network_clusters(G):
    print("\n--- Running Graph Clustering (Louvain) ---")
    # Louvain algorithm network ko optimized sub-networks mein tod dega
    communities = louvain_communities(G, weight='avg_latency_ms')
    
    print(f"✅ Network partitioned into {len(communities)} optimized clusters.")
    
    # Har cluster mein kitne nodes hain wo nikalne ke liye
    cluster_info = {}
    for i, community in enumerate(communities):
        cluster_info[f"Cluster_{i+1}"] = len(community)
            
    return cluster_info

def optimize_network_path(G, source_node, target_node):
    print(f"\nFinding optimized path from {source_node} to {target_node}...")
    try:
        best_path = nx.shortest_path(G, source=source_node, target=target_node, weight='avg_latency_ms')
        total_latency = nx.shortest_path_length(G, source=source_node, target=target_node, weight='avg_latency_ms')
        print(f"✅ Optimized Path Found: {' -> '.join(best_path)}")
        return best_path, total_latency
    except nx.NetworkXNoPath:
        return None, None

# Run karne ka main block (Testing ke liye)
if __name__ == "__main__":
    graph = load_cloud_topology()
    clusters = partition_network_clusters(graph)
    optimize_network_path(graph, source_node="service_1", target_node="service_50")