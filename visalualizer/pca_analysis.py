import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from calculate_std_mean import calculate_statistics
from helper import get_ids_by_partition

plt.rcParams['font.family'] = 'Palatino Linotype'

file = "data/txt_files/list_attr_celeba.txt"

df_id_training = get_ids_by_partition(0)
df_std_training, df_mean_training = calculate_statistics(file, df_id_training, balanced=False)
df_std_training.to_csv("attr_std_training.csv")
df_mean_training.to_csv("attr_mean_training.csv")

df_id_validation = get_ids_by_partition(1)
df_std_validation, df_mean_validation = calculate_statistics(file, df_id_validation, balanced=False)
df_std_validation.to_csv("attr_std_validation.csv")
df_mean_validation.to_csv("attr_mean_validation.csv")


training_dfs = [df_std_training.drop(columns="n"), df_mean_training.drop(columns="n")]
validation_dfs = [df_std_validation.drop(columns="n"), df_mean_validation.drop(columns="n")]


files = ["Standard Deviation", "Mean"]
# training_files = [df_mean_training, df_std_training]
# validation_files = [df_mean_validation, df_std_validation]
colors = ["blue", "red"]

# iterating through both files (type of statistical measurement)
for idx, name in enumerate(files):
    df_training = training_dfs[idx]
    df_validation = validation_dfs[idx]
    pca = PCA()
    # fitting on training data
    pca.fit(df_training)
    # transforming to validation data
    pca_data = pca.transform(df_validation)
    # getting the number of explained variation
    percentage_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(percentage_var) + 1)]

    pca_df = pd.DataFrame(pca_data, index=df_validation.index, columns=labels)

    fig1 = plt.figure(figsize=(4, 4), dpi=80)
    plt.bar(x=range(1, len(percentage_var) + 1), height=percentage_var, tick_label=labels, color=colors[idx])
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title("Scree Plot " + name)
    plt.xticks(fontsize=8, rotation=90)
    plt.show()

    fig2 = plt.figure(figsize=(4, 4), dpi=80)
    plt.scatter(pca_df.PC1, pca_df.PC2, 5, c=colors[idx])
    plt.title("PCA " + name)
    plt.xlabel("PC1 - {0}%".format(percentage_var[0]))
    plt.ylabel("PC2 - {0}%".format(percentage_var[1]))
    plt.show()
    fig2.savefig("resources/PCA_"+name+".pdf")

