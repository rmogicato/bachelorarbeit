import pandas as pd
import numpy as np


validation_file = "../data/txt_files/list_attr_celeba.txt"
mean_file = "ground_truth_operations/mean.csv"
std_file = "_DEPRECATED/ground_truth_operations/std.csv"
image_threshold = 5

df_id = pd.read_csv("../data/txt_files/identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
df_raw = pd.read_csv(validation_file, header=0, sep='\s+')
df_raw = df_raw.merge(df_id, how="left", on="Image")
df_mean = pd.read_csv(mean_file, header=0, index_col=0).reset_index()
df_std = pd.read_csv(std_file, header=0, index_col=0).reset_index()

df_mean = df_mean.drop(df_mean[df_mean.n < image_threshold].index)
df_std = df_std.drop(df_std[df_std.n < image_threshold].index)

# taking all attributes besides the last two (Image and Id)
attributes = df_raw.columns.values.tolist()[1:-1]
print(attributes)
# only keep pictures and their detected attributes of identities that are in mean_file and std_file
ids = df_mean["index"].tolist()
df_attributes = df_raw[df_raw["Id"].isin(ids)]
print(df_attributes)

# asserting that both indexes are equal i.d that the identities are the same
assert len(df_mean.index) == len(df_std.index) and sorted(df_mean.index) == sorted(df_std.index)

# split identities by gender
male_ids = []
female_ids = []
nb_ids = []

for identity in ids:
    gender_mean = df_mean.loc[df_mean["index"] == identity].Male.values[0]

    if gender_mean >= 0.3:
        male_ids.append(identity)
    elif gender_mean <= -0.3:
        female_ids.append(identity)
    else:
        nb_ids.append(identity)

print("male:", len(male_ids))
print("female:", len(female_ids))
print("non-binary:", nb_ids)

"""
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_attributes.loc[df_attributes["Id"].isin(nb_ids)].sort_values(by="Id"))

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_mean.loc[df_mean["index"].isin(nb_ids)])
"""
df_mean_male = df_mean.loc[df_mean["index"].isin(male_ids)]
df_std_male = df_std.loc[df_std["index"].isin(male_ids)]

df_mean_female = df_mean.loc[df_mean["index"].isin(female_ids)]
df_std_female = df_std.loc[df_std["index"].isin(female_ids)]


def get_factors_df(fac_mean, fac_std):
    dictionary = {}
    for a in attributes:
        mean = np.round(np.mean(fac_mean[a].values), 2)
        std = np.round(np.mean(fac_std[a].values), 2)
        inverse_std = (1 - std if std < 1 else 0)
        factor_m = inverse_std ** 2 * mean
        dictionary[a] = {"mean": mean, "std": std, "factor": factor_m}
    return dictionary


dict_m = get_factors_df(df_mean_male, df_std_female)
dict_f = get_factors_df(df_mean_female, df_std_female)

df_male = pd.DataFrame.from_dict(dict_m).transpose()
df_female = pd.DataFrame.from_dict(dict_f).transpose()

# print(df_male.sort_values(by="factor"))
# print(df_female.sort_values(by="factor"))

attr = df_raw.loc[df_raw.Id == 6101]
print(attr)