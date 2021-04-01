import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from altair import *
import matplotlib.backends.backend_pdf

stat_type = "std"
file_training = "calculated_csv/std_automatic_training.csv"
file_validation = "calculated_csv/std_automatic_validation.csv"

df_raw_training = pd.read_csv(file_training, header=0, index_col=0)
df_raw_validation = pd.read_csv(file_validation, header=0, index_col=0)
df_training = df_raw_training.drop(columns="n")
df_validation = df_raw_validation.drop(columns="n")

attributes = df_training.columns

# scaling data?
pca = PCA()
pca.fit(df_training)
pca_data = pca.transform(df_validation)
percentage_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ["PC" + str(x) for x in range(1, len(percentage_var) + 1)]

pca_df = pd.DataFrame(pca_data, index=df_validation.index, columns=labels)

# creating a scatterplot for each attribute
for attribute in attributes:

    # scatterplot with color bar, coded on the range [-1, 1] according to the attribute
    fig, ax = plt.subplots()
    im = ax.scatter(pca_df.PC1, pca_df.PC2, c=df_validation[attribute], cmap=plt.cm.get_cmap('plasma'), s=3)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title(attribute)
    im.set_clim(df_validation[attribute].min(), df_validation[attribute].max())

    plt.title("PCA " + stat_type)
    plt.xlabel("PC1 - {0}%".format(percentage_var[0]))
    plt.ylabel("PC2 - {0}%".format(percentage_var[1]))

    # adding figure to figures array, to print to pdfs later
    fig = plt.gcf()
    fig.savefig("pca_by_attribute_validation/"+"PCA_"+stat_type+"_"+attribute+".png")
    plt.show()