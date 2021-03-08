import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from altair import *
import matplotlib.backends.backend_pdf

files = ["std_automatic.csv", "mean_automatic.csv"]

for idx, file in enumerate(files):
    df_raw = pd.read_csv(file, header=0, index_col=0)

    df = df_raw.drop(columns="n")
    attributes = df.columns

    # scaling data?
    pca = PCA()
    pca.fit(df)
    pca_data = pca.transform(df)
    percentage_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(percentage_var) + 1)]

    pca_df = pd.DataFrame(pca_data, index=df.index, columns=labels)

    figures = []

    # creating a scatterplot for each attribute
    for attribute in attributes:

        # scatterplot with color bar, coded on the range [-1, 1] according to the attribute
        fig, ax = plt.subplots()
        im = ax.scatter(pca_df.PC1, pca_df.PC2, c=df[attribute], cmap=plt.cm.get_cmap('plasma'), s=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_title(attribute)
        im.set_clim(df[attribute].min(), df[attribute].max())

        plt.title("PCA " + file[:-4])
        plt.xlabel("PC1 - {0}%".format(percentage_var[0]))
        plt.ylabel("PC2 - {0}%".format(percentage_var[1]))

        # adding figure to figures array, to print to pdfs later
        fig = plt.gcf()
        figures.append(fig)
        plt.show()

    # saves all plots to pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("PCA_" + file[:-4] +"_by_attributes" +".pdf")

    for fig in figures:
        pdf.savefig(fig)
    pdf.close()
