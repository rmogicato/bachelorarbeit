import pandas as pd
import numpy as np
from math import comb, log
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ids_clustered = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Arc_id"])
ids_gt = pd.read_csv("data/txt_files/identity_CelebA.txt", sep='\s+', names=["Image", "Gt_id"])
partition = pd.read_csv("data/txt_files/list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])

partition = partition.loc[partition.Partition == 2]
ids_gt = ids_gt.merge(partition, on="Image").drop(columns="Partition")
merged = ids_gt.merge(ids_clustered, on="Image")

clusters_id = np.unique(merged.Arc_id.values)
n_correct = 0

# iterate over each cluster and calculate its purity
for i in clusters_id:
    # merged clusters with classes for identity i
    clustered_df = merged.loc[merged["Arc_id"] == i]
    # classes in that cluster
    ids = clustered_df.Gt_id.values
    counts = np.bincount(ids)
    # most frequent class
    dominant_id = counts.argmax()
    # number of correctly assigned classes
    n = counts[dominant_id]
    n_correct += n


purity = n_correct/merged.shape[0]
print("purity: ", purity)

actual_labels = merged.Gt_id.values
predicted_labels = merged.Arc_id.values
ri_score = adjusted_rand_score(actual_labels, predicted_labels)
nmi = normalized_mutual_info_score(actual_labels, predicted_labels)
print("rand index: ", ri_score)
print("NMI score:",  nmi)

