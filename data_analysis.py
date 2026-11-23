import pandas as pd

alibaba = pd.read_csv("DATASET/alibaba_microservices_large.csv")

print("Total connections:", len(alibaba))
print("Unique source nodes:", alibaba["source_node"].nunique())
print("Unique target nodes:", alibaba["target_node"].nunique())
print("Bottleneck counts:")
print(alibaba["bottleneck_flag"].value_counts())