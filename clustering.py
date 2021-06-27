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

df_arcface = pd.read_csv("arc_face/arcface_testing_v2.csv")
df_arcface = df_arcface.rename(columns={"Unnamed: 0": "Image"})

df_identities = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
images = df_arcface["Image"]


def suffixizer(x):
    x = x[:-3]
    return x + "jpg"


images = images.apply(suffixizer)

ids_df = df_identities.loc[df_identities["Image"].isin(images)]
ids = ids_df.Id.tolist()
bins = np.bincount(ids)
data = []
for b in bins:
    if b != 0:
        data.append(b)

bins = np.array(data)
print(bins)

unique_ids = len(np.unique(ids))
print(unique_ids)
est_ids = (int(unique_ids - unique_ids * 0.2), int(unique_ids + unique_ids * 0.4))

df_cosines = df_arcface.drop(columns="Image")
cosines = df_cosines.values
model = sklearn.cluster.AgglomerativeClustering(memory="cluster", affinity="euclidean", linkage="complete",
                                                compute_full_tree=True)

"""
model.n_clusters = 1163
model.fit(cosines)
labels = np.array(model.labels_)
print(labels.shape)

df_clustered = pd.DataFrame(data=labels, index=images.values)
df_clustered = df_clustered.reset_index()
df_clustered.to_csv("ids/arcface_testing_euclidean_ids.txt", index=False, sep=" ", header=False)


"""
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
    p = str(round((i-est_ids[0]) / (est_ids[1]-est_ids[0]) * 100, 0))
    sys.stdout.write("\r Calculating silhouette score. Progress: " + p + "%")
    sys.stdout.flush()
    i += 1


for key in s_scores:
    x = key
    y = s_scores[key]
    plt.scatter(x, y, color="b")

sorted_attribute_std = {k: v for k, v in sorted(s_scores.items(), key=lambda item: item[1])}
print(sorted_attribute_std)
plt.show()


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

"""
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))
plt.show()
"""

"""
cluster.fit_predict(cosines)
df_test = df_arcface.head(100)
print(df_test)
df_test["cluster"] = cluster.labels_
print(cluster.labels_)
print(df_test)

print([{'node_id': next(itertools.count(cosines.shape[0])), 'left': x[0], 'right':x[1]} for x in cluster.children_])
"""
