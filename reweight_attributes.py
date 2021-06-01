import numpy as np
import pandas as pd
import sys


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

        # 40 attributes
        factors = inverse_std**2 * mean
        factors_series = pd.Series(factors)
        row_df = pd.DataFrame([factors], index=[i])

        # append to df
        df_factors = pd.concat([df_factors, row_df])

    return df_factors


def reweight_attributes(validation_file, mean_file, std_file, df_id):

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

    # only keep pictures and their detected attributes of identities that are in the partition
    ids = np.unique(df_id.Id.values)

    # index is the id, rows 0-39 are the 40 attributes
    df_factors = get_unbalanced_factors(df_mean, df_std, ids).reset_index()

    # rename the columns
    columns = df_raw.columns.values.tolist()[:-2]
    columns.insert(0, "Id")
    df_factors.columns = columns

    # changing Id to int for easy comparison
    df_factors.Id = df_factors.Id.astype(int)
    print(df_factors)

    print("\n\n\n attributes", df_raw)

    df_corrected = reweigh_unbalanced(ids, df_raw, df_factors)

    return df_corrected


def reweigh_unbalanced(ids, df_raw, df_factors):
    df_corrected = pd.DataFrame()
    for p, i in enumerate(ids):
        p = str(round(p / len(ids) * 100, 0))
        sys.stdout.write("\r Reweighting file. Progress: " + p + "%")
        sys.stdout.flush()

        df_data = df_raw.loc[df_raw["Id"] == i]
        image = df_data.Image

        df_data = df_data.drop(columns=["Image", "Id"])

        attribute_factors = df_factors.loc[df_factors["Id"] == i]
        factors = np.array(attribute_factors.drop(columns="Id").values[0])

        df_new = df_data.apply(lambda x: np.add(factors, x), raw=True, axis=1)
        df_new["Image"] = image
        df_corrected = df_corrected.append(df_new)
    return df_corrected


def reweigh_balanced(ids, df_raw, df_factors):
    df_corrected = pd.DataFrame()
    for p, i in enumerate(ids):
        p = str(round(p / len(ids) * 100, 0))
        sys.stdout.write("\r Reweighting file. Progress: " + p + "%")
        sys.stdout.flush()

        df_data = df_raw.loc[df_raw["Id"] == i]
        image = df_data.Image

        df_data = df_data.drop(columns=["Image", "Id"])

        attribute_factors = df_factors.loc[df_factors["Id"] == i]
        factors = np.array(attribute_factors.drop(columns="Id").values[0])

        df_new = df_data.apply(lambda x: np.add(factors, x), raw=True, axis=1)
        df_new["Image"] = image
        df_corrected = df_corrected.append(df_new)
    return df_corrected


validation_file = "extractions/new_testing_AFFACT2.txt"
mean_file = "extractions/means/mean_gt_testing_AFFACT2.txt"
std_file = "extractions/stds/std_gt_testing_AFFACT2.txt"
df_id = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Id"])


df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
partition = pd.read_csv("list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
images = partition.loc[partition.Partition == 2].Image.values
df_id = df_id.loc[df_id.Image.isin(images)]

corrected = reweight_attributes(validation_file, mean_file, std_file, df_id)
corrected = corrected.sort_values(by="Image")


print(corrected)
corrected.to_csv("extractions/reweighed/gt_testing_reweighed_square_mean_AFFACT2.txt")