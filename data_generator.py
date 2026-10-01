import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 1. Yatharth ke GNN Model ke liye (Topology Data Generator)
def generate_topology(num_rows=2000):
    services = [f"service_{i}" for i in range(1, 101)]
    data = []
    for _ in range(num_rows):
        src = random.choice(services)
        tgt = random.choice(services)
        if src == tgt: continue
        protocol = random.choice(['tcp', 'http', 'grpc', 'udp'])
        latency = round(np.random.gamma(2, 20), 2) # Realistic latency distribution
        transfer = round(np.random.uniform(10, 8000), 2)
        error_rate = round(np.random.exponential(0.5), 2)
        
        # Bottleneck logic: Agar latency 100ms se zyada ya error 2% se zyada hai
        bottleneck = 1 if latency > 100 or error_rate > 2.0 else 0
        data.append([src, tgt, protocol, latency, transfer, error_rate, bottleneck])
    
    df = pd.DataFrame(data, columns=['source_node', 'target_node', 'protocol', 'avg_latency_ms', 'data_transfer_mb', 'error_rate_percent', 'bottleneck_flag'])
    df.to_csv("alibaba_microservices_large.csv", index=False)
    print(f"Topology data saved with {num_rows} rows!")

# 2. Akshat ke LSTM Model ke liye (Telemetry/Time-Series Generator)
def generate_telemetry(num_rows=5000):
    machines = [f"node_{i}" for i in range(1001, 1051)]
    data = []
    start_time = datetime(2026, 9, 27, 10, 0, 0)
    for i in range(num_rows):
        m_id = random.choice(machines)
        # Data har 5 minute ke gap mein generate hoga
        timestamp = start_time + timedelta(minutes=5*i)
        cpu = round(np.random.uniform(5, 99), 1)
        mem = round(np.random.uniform(10, 95), 1)
        disk = round(np.random.uniform(100, 15000), 0)
        rx = round(np.random.uniform(1000, 10000000), 0)
        tx = round(np.random.uniform(1000, 10000000), 0)
        data.append([timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'), m_id, cpu, mem, disk, rx, tx])
        
    df = pd.DataFrame(data, columns=['timestamp', 'machine_id', 'cpu_usage_percent', 'memory_usage_percent', 'disk_io_rate', 'network_rx_bytes', 'network_tx_bytes'])
    df.to_csv("google_cluster_large.csv", index=False)
    print(f"Telemetry data saved with {num_rows} rows!")

# 3. Harsh ke FinOps/XGBoost Model ke liye (Financial Data Generator)
def generate_finops(num_rows=2000):
    providers = ['AWS', 'GCP', 'Azure']
    regions = ['us-east-1', 'europe-west3', 'ap-south-1', 'us-west-2']
    resources = ['Compute', 'Database', 'Storage', 'Cache']
    families = ['c5.xlarge', 'rds.m5.large', 'e2-medium', 'Premium_LRS', 't3.micro']
    data = []
    start_date = datetime(2026, 9, 1)
    
    for i in range(num_rows):
        date = (start_date + timedelta(days=random.randint(0, 60))).strftime('%Y-%m-%d')
        prov = random.choice(providers)
        reg = random.choice(regions)
        res = random.choice(resources)
        fam = random.choice(families)
        hours = random.choice([12, 24, 720]) # Daily or Monthly usage
        cost = round(np.random.uniform(0.5, 50.0), 2)
        recom = random.choice(['None', 'Scale Down', 'Delete Unattached', 'Right-size', 'Upgrade'])
        data.append([date, prov, reg, res, fam, hours, cost, recom])
        
    df = pd.DataFrame(data, columns=['date', 'cloud_provider', 'region', 'resource_type', 'instance_family', 'usage_hours', 'cost_usd', 'optimization_recommendation'])
    df.to_csv("finops_cloud_cost_large.csv", index=False)
    print(f"FinOps data saved with {num_rows} rows!")

# Script Run Karna
if __name__ == "__main__":
    print("Generating large CSV datasets for GraphOps AI...")
    generate_topology(3000)   # Yatharth ke liye 3000 rows
    generate_telemetry(8000)  # Akshat ke liye 8000 rows
    generate_finops(3000)     # Harsh ke liye 3000 rows
    print("All CSV files generated successfully in your folder!")