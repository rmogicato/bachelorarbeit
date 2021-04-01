import numpy as np
import pandas as pd

validation_file = "data-AFFACT2/extracted_attributes_validation_AFFACT2.txt"
mean_file = "data-AFFACT2/mean_automatic_validation.calculated_csv"
std_file = "data-AFFACT2/std_automatic_validation.calculated_csv"
image_threshold = 0

df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
df_raw = pd.read_csv(validation_file, header=0, index_col=0)
df_raw = df_raw.merge(df_id, how="left", on="Image")
df_mean = pd.read_csv(mean_file, header=0, index_col=0).reset_index()
df_std = pd.read_csv(std_file, header=0, index_col=0).reset_index()

df_mean = df_mean.drop(df_mean[df_mean.n < image_threshold].index)
df_std = df_std.drop(df_std[df_std.n < image_threshold].index)

# taking all attributes besides the last two (Image and Id)
attributes = df_raw.columns.values.tolist()[:-2]

# only keep pictures and their detected attributes of identities that are in mean_file and std_file
ids = df_mean["index"].tolist()
df_attributes = df_raw[df_raw["Id"].isin(ids)]


# asserting that both indexes are equal i.d that the identities are the same
assert len(df_mean.index) == len(df_std.index) and sorted(df_mean.index) == sorted(df_std.index)

df_factors = pd.DataFrame()

# iterate through each id and get it's mean and std for each attribute
for i in ids:
    mean = np.array(df_mean.loc[df_mean["index"] == i].values[0][1:41])
    std = np.array(df_std.loc[df_mean["index"] == i].values[0][1:41])
    inverse_std = []
    for s in std:
        if s < 1:
            inverse_std.append(1-s)
        else:
            inverse_std.append(0)
    inverse_std = np.array(inverse_std)

    # this determines the weighting factor
    factors = inverse_std**2 * mean
    factors = np.append(factors, i)

    # append to df
    df_factors = df_factors.append(pd.Series(factors), ignore_index=True)

# rename the columns
columns = df_raw.columns.values.tolist()[:-2]
columns.append("Id")
df_factors.columns = columns

# changing Id to int for easy comparison
df_factors.Id = df_factors.Id.astype(int)
print(df_factors)

df_corrected = pd.DataFrame()


for i in ids:
    df_data = df_attributes.loc[df_attributes["Id"] == i]
    image = df_data.Image
    df_data = df_data.drop(columns=["Image", "Id"])
    factors = np.array(df_factors.loc[df_factors["Id"] == i].drop(columns="Id").values[0])

    df_new = df_data.apply(lambda x: np.add(factors, x), raw=True, axis=1)
    df_new["Image"] = image
    df_corrected = df_corrected.append(df_new)

print(df_corrected)

df_corrected.to_csv("data-AFFACT2/extracted_attributes_validation_corrected.txt")


