import numpy as np
import pandas as pd


def get_unbalanced_factors(df_mean, df_std, ids):
    df_factors = pd.DataFrame()

    # iterate through each id and get it's mean and std for each attribute
    for i in ids:
        mean = np.array(df_mean.loc[df_mean["index"] == i].values[0][1:41])
        std = np.array(df_std.loc[df_mean["index"] == i].values[0][1:41])
        inverse_std = []
        for s in std:
            if s < 1:
                inverse_std.append(1 - s)
            else:
                inverse_std.append(0)
        inverse_std = np.array(inverse_std)

        # this determines the weighting factor

        factors = inverse_std ** 3 * np.sign(mean)
        factors = np.append(factors, i)

        # append to df
        df_factors = df_factors.append(pd.Series(factors), ignore_index=True)

    return df_factors


def get_balanced_factors(df_mean, df_std, ids):
    df_factors = pd.DataFrame()

    # iterate through each id and get it's mean and std for each attribute
    for i in ids:
        mean = np.array(df_mean.loc[df_mean["index"] == i].values[0][1:41])
        std = np.array(df_std.loc[df_mean["index"] == i].values[0][1:41])
        inverse_std = []
        for s in std:
            if s < 1:
                inverse_std.append(1 - s)
            else:
                inverse_std.append(0)
        inverse_std = np.array(inverse_std)

        # this determines the weighting factor

        factors = inverse_std ** 3 * np.sign(mean)
        factors = np.append(factors, i)

        # append to df
        df_factors = df_factors.append(pd.Series(factors), ignore_index=True)
    return df_factors


validation_file = "extractions/new_extract_validation_AFFACT1.txt"
mean_file = "extractions/means/mean_new_extract_validation_AFFACT1.txt"
std_file = "extractions/stds/std_new_extract_validation_AFFACT1.txt"

# 0 --> training, 1 --> validation, 2 --> testing
partitions = [1]

"""
# linear approach:
factors = inverse_std * mean

# squared approach:
factors = inverse_std**2 * mean

# sigmoid approach:
factors = 1 / (1 + np.exp(x=-inverse_std)

# crazy approach
factors = np.sign(mean) / (1 + ((inverse_std / (1 - inverse_std)) ** -np.e))

# crazy2 approach marginally bettter than crazy 1
(1 / (1 + ((inverse_std / (1 - inverse_std)) ** -np.e)))*mean

"""


df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
df_partition = pd.read_csv("list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
df_partition = df_partition.merge(df_id, how="left", on="Image")

if validation_file == "list_attr_celeba.txt":
    df_raw = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
    cols = df_raw.columns.tolist()
    cols = cols[1:] + [cols[0]]
    df_raw = df_raw[cols]
else:
    df_raw = pd.read_csv(validation_file, header=0, index_col=0)

df_raw = df_raw.merge(df_id, how="left", on="Image")
df_mean = pd.read_csv(mean_file, header=0, index_col=0).reset_index()
df_std = pd.read_csv(std_file, header=0, index_col=0).reset_index()

# taking all attributes besides the last two (Image and Id)
attributes = df_raw.columns.values.tolist()[:-2]

# only keep pictures and their detected attributes of identities that are in the partition
ids = np.unique(df_partition.loc[df_partition["Partition"].isin(partitions)].Id.values)
df_attributes = df_raw[df_raw["Id"].isin(ids)]

df_factors = get_unbalanced_factors(df_mean, df_std, ids)

# rename the columns
columns = df_raw.columns.values.tolist()[:-2]
columns.append("Id")
df_factors.columns = columns

# changing Id to int for easy comparison
df_factors.Id = df_factors.Id.astype(int)
print(df_factors)

print("\n\n\n attributes", df_attributes)

df_corrected = pd.DataFrame()

for i in ids:
    df_data = df_attributes.loc[df_attributes["Id"] == i]
    image = df_data.Image
    df_data = df_data.drop(columns=["Image", "Id"])
    attribute_factors = df_factors.loc[df_factors["Id"] == i]
    factors = np.array(attribute_factors.drop(columns="Id").values[0])
    df_new = df_data.apply(lambda x: np.add(factors, x), raw=True, axis=1)
    df_new["Image"] = image
    df_corrected = df_corrected.append(df_new)

print(df_corrected)

df_corrected.to_csv("extractions/reweighed/validation_reweighed_cube_sign.txt")


