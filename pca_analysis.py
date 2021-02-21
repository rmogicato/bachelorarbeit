import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_raw = pd.read_csv("mean.csv", header=0, index_col=0)

# todo: sort out subjects with an n below a certain threshold
df = df_raw.drop(columns="n")

# scaling data?
pca = PCA()
pca.fit(df)
pca_data = pca.transform(df)
percentage_var = np.round(pca.explained_variance_ratio_, decimals=4)
labels = ["PC" + str(x) for x in range(1, len(percentage_var) + 1)]

plt.bar(x=range(1, len(percentage_var) + 1), height=percentage_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Component")
plt.title("Scree Plot")
plt.show()


pca_df = pd.DataFrame(pca_data, index=df.index, columns=labels)
print(pca_df)
plt.scatter(pca_df.PC1, pca_df.PC2, 1)
plt.title("PCA")
plt.xlabel("PC1 - {0}%".format(percentage_var[0] * 100))
plt.ylabel("PC2 - {0}%".format(percentage_var[1] * 100))
plt.show()

"""
dbscan = DBSCAN(eps=10, min_samples=2)
clusters = dbscan.fit_predict(pca_df)

# plot the cluster assignments
plt.scatter(pca_df.PC1, pca_df.PC2, 1, c=clusters, cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
"""