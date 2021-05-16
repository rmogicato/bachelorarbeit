import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

files = ["mean_automatic.calculated_csv", "std_automatic.calculated_csv"]
colors = ["blue", "red"]

for idx, file in enumerate(files):
    df_raw = pd.read_csv(file, header=0, index_col=0)

    # dropping all rows where there are fewer than 5 images for a person
    df_raw = df_raw.drop(df_raw[df_raw.n < 5].index)
    df = df_raw.drop(columns="n")
    print(df)

    # scaling data?
    pca = PCA()
    pca.fit(df)
    pca_data = pca.transform(df)
    percentage_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(percentage_var) + 1)]

    plt.bar(x=range(1, len(percentage_var) + 1), height=percentage_var, tick_label=labels, color=colors[idx])
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title("Scree Plot " + file[:-4])
    plt.xticks(fontsize=8, rotation=90)
    plt.show()

    pca_df = pd.DataFrame(pca_data, index=df.index, columns=labels)
    print(pca_df)
    plt.scatter(pca_df.PC1, pca_df.PC2, 1, c=colors[idx])
    plt.title("PCA " + file[:-4])
    plt.xlabel("PC1 - {0}%".format(percentage_var[0]))
    plt.ylabel("PC2 - {0}%".format(percentage_var[1]))
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