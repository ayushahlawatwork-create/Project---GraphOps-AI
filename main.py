from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Naye function (partition_network_clusters) ko import kar rahe hain
from optimization_engine import load_cloud_topology, optimize_network_path, partition_network_clusters

app = FastAPI(title="GraphOps AI - System Orchestrator", version="1.0")

print("Initializing Graph Database in memory...")
graph = load_cloud_topology()

class RouteRequest(BaseModel):
    source_node: str
    target_node: str

@app.get("/")
def read_root():
    return {"status": "GraphOps AI Backend is Live and Running!"}

# Purana Routing Endpoint
@app.post("/api/v1/optimization/routes")
def calculate_optimized_route(request: RouteRequest):
    path, latency = optimize_network_path(graph, request.source_node, request.target_node)
    
    if path is None:
        raise HTTPException(status_code=404, detail="In dono nodes ke beech koi rasta nahi hai.")
        
    return {
        "status": "success",
        "source": request.source_node,
        "target": request.target_node,
        "optimized_path": path,
        "total_estimated_latency_ms": latency
    }

# NAYA: Clustering Endpoint
@app.get("/api/v1/optimization/clusters")
def get_network_clusters():
    # Aapka Louvain clustering function call hoga
    cluster_data = partition_network_clusters(graph)
    
    return {
        "status": "success",
        "total_clusters": len(cluster_data),
        "cluster_distribution": cluster_data
    }