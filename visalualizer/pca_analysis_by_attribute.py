import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Palatino Linotype'

# either "Mean" or "Standard Deviation"
stat_type = "Mean"
# execute pca_analysis.py to get these two files
file_training = "attr_mean_training.csv"
file_validation = "attr_mean_validation.csv"

df_raw_training = pd.read_csv(file_training, header=0, index_col=0)
df_raw_validation = pd.read_csv(file_validation, header=0, index_col=0)
df_training = df_raw_training.drop(columns="n")
df_validation = df_raw_validation.drop(columns="n")

attributes = df_training.columns
attributes_clean = []

for a in attributes:
    attributes_clean.append(a.replace("_", " "))

# scaling data?
pca = PCA()
pca.fit(df_training)
pca_data = pca.transform(df_validation)
percentage_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ["PC" + str(x) for x in range(1, len(percentage_var) + 1)]

pca_df = pd.DataFrame(pca_data, index=df_validation.index, columns=labels)


for i, attribute in enumerate(attributes):

    # scatterplot with color bar, coded on the range [-1, 1] according to the attribute
    fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
    im = ax.scatter(pca_df.PC1, pca_df.PC2, c=df_validation[attribute], cmap=plt.cm.get_cmap('copper'), s=5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title(attributes_clean[i])
    im.set_clim(df_validation[attribute].min(), df_validation[attribute].max())

    plt.title("PCA " + stat_type)
    plt.xlabel("PC1 - {0}%".format(percentage_var[0]))
    plt.ylabel("PC2 - {0}%".format(percentage_var[1]))

    fig.savefig("../pca_by_attribute_validation/"+"PCA_"+stat_type+"_"+attribute+".pdf")
    plt.show()