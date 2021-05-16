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




df_arcface = pd.read_csv("arc_face/arcface_validation.csv")
df_arcface = df_arcface.rename(columns={"Unnamed: 0":"Image"})


df_identities = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
images = df_arcface["Image"]


def suffixizer(x):
    x = x[:-3]
    return x+"jpg"


images = images.apply(suffixizer)

res = df_identities.loc[df_identities["Image"].isin(images)]
ids = res.Id.tolist()
unique_ids = len(np.unique(ids))
est_ids = (int(unique_ids - (unique_ids*0.1)), int(unique_ids + (unique_ids*0.05)))
print(est_ids)

df_cosines = df_arcface.drop(columns="Image")
cosines = df_cosines.values


model = sklearn.cluster.AgglomerativeClustering(memory="cluster", affinity="cosine", linkage="average", compute_full_tree=True)
# cluster.fit(cosines)

visualizer = KElbowVisualizer(model, k=est_ids)
visualizer.fit(cosines)
visualizer.show()

value = visualizer.elbow_value_
print(value)

model.n_clusters = value
model.fit(cosines)

labels = np.array(model.labels_)
print(labels)
count = np.bincount(labels)
print(count)
print(count.shape)
print("median: ", np.median(count))



"""
pca = PCA(n_components=2)
X_principal = pca.fit_transform(df_cosines.head(100))
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

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

