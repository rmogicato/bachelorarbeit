from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import sklearn.cluster
import itertools
from scipy.cluster.hierarchy import dendrogram
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import KElbowVisualizer
import sys
from sklearn.metrics import silhouette_samples, silhouette_score

# np.set_printoptions(threshold=sys.maxsize)

# set to extractions of arcface
df_arcface = pd.read_csv("ArcFace/arcface_testing_v2.csv")
df_arcface = df_arcface.rename(columns={"Unnamed: 0": "Image"})

df_identities = pd.read_csv("data/txt_files/identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
images = df_arcface["Image"]


def suffixizer(x):
    x = x[:-3]
    return x + "jpg"


images = images.apply(suffixizer)

ids_df = df_identities.loc[df_identities["Image"].isin(images)]
ids = ids_df.Id.tolist()

unique_ids = len(np.unique(ids))
print(unique_ids)

# we estimate the ids based on the number of unique ids of the images
# this step only exists to reduce the computing time and limit where we look for the maximal silhouette score
# we can search on any interval, like (0, 2000)
est_ids = (int(unique_ids - unique_ids * 0.5), int(unique_ids + unique_ids * 0.5))

df_cosines = df_arcface.drop(columns="Image")
cosines = df_cosines.values
model = sklearn.cluster.AgglomerativeClustering(memory="cluster", affinity="cosine", linkage="complete",
                                                compute_full_tree=True)


def silhouette(model, n_clusters):
    model.n_clusters = n_clusters
    cluster_label = model.fit_predict(cosines)
    silhouette_avg = silhouette_score(cosines, cluster_label)
    return n_clusters, silhouette_avg


i = est_ids[0]
s_scores = {}

while i <= est_ids[1]:
    n_clusters, silhouette_avg = silhouette(model, i)
    s_scores[n_clusters] = silhouette_avg
    dif = est_ids[1]-est_ids[0]
    if dif == 0:
        dif = 1
    p = str(round((i-est_ids[0]) / dif * 100, 0))
    sys.stdout.write("\r Calculating silhouette score. Progress: " + p + "%")
    sys.stdout.flush()
    i += 1


for key in s_scores:
    x = key
    y = s_scores[key]
    plt.scatter(x, y, color="b")

sorted_silhouette = {k: v for k, v in sorted(s_scores.items(), key=lambda item: item[1])}
max_score = list(sorted_silhouette)[-1]
print("\nMaximum silhouette score with number of clusters = ", max_score)

model.n_clusters = max_score
model.fit(cosines)
labels = np.array(model.labels_)


df_clustered = pd.DataFrame(data=labels, index=images.values)
df_clustered = df_clustered.reset_index()

# uncomment to save file
# df_clustered.to_csv("ids/arcface_testing_ids.txt", index=False, sep=" ", header=False)


plt.show()

# deprecated code for Elbow method, as this method was very unsuccessful
"""
visualizer = KElbowVisualizer(model, k=est_ids)
visualizer.fit(cosines)
visualizer.show()

value = visualizer.elbow_value_
print(value)

model.n_clusters = value
model.fit(cosines)

labels = np.array(model.labels_)
count = np.bincount(labels)
print(count)
print(count.shape)
print("stds: ", np.std(bins), np.std(count))
"""
