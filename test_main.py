from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "GraphOps AI Backend is Live and Running!"}

def test_get_network_clusters():
    response = client.get("/api/v1/optimization/clusters")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total_clusters" in data
    assert "cluster_distribution" in data

def test_calculate_optimized_route():
    payload = {"source_node": "service_1", "target_node": "service_50"}
    response = client.post("/api/v1/optimization/routes", json=payload)
    
    assert response.status_code in [200, 404]