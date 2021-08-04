import numpy as np
import pandas as pd
import sys

from helper import calculate_distribution

"""
this function calculates weights based on the mean, std, ids and reweight formula.
The ids should correspond to the index of df_mean and df_std
"""


def get_weights(df_mean, df_std, ids, reweight_formula):
    df_factors = pd.DataFrame()
    # iterate through each id and get it's mean and std for each attribute

    formulas = {
        "square_mean": lambda i_s, m: inverse_std ** 2 * m,
        "cube_mean": lambda i_s, m: inverse_std ** 3 * m,
        "square_sign": lambda i_s, m: inverse_std ** 2 * np.sign(m),
        "cube_sign": lambda i_s, m: inverse_std ** 3 * np.sign(m),
        "quad_mean": lambda i_s, m: inverse_std ** 4 * m,
        "quad_sign": lambda i_s, m: inverse_std ** 4 * np.sign(m),
        "quint_sign": lambda i_s, m: inverse_std ** 5 * np.sign(m),
        "quint_mean": lambda i_s, m: inverse_std ** 5 * m,
        "cosine_mean": lambda i_s, m: ((-np.cos(i_s * np.pi) + 1) / 2) * np.sign(m),
        "sigmoid1": lambda i_s, m: 1 / (1 + np.e ** (-10 * (i_s - 0.5))) * np.sign(m),
        "sigmoid2": lambda i_s, m: 1 / (1 + np.e ** (-10 * (i_s - 0.6))) * np.sign(m),
    }

    if reweight_formula not in list(formulas.keys()):
        raise ValueError("Invalid formula, please choose on of these formulas: %s" % list(formulas.keys()))

    for i in ids:
        # taking mean and std of the first 40 entries (attributes)
        mean = np.array(df_mean.loc[i].tolist()[0:40])
        std = np.array(df_std.loc[i].tolist()[0:40])

        inverse_std = []

        # calculating 1-std and making sure that there are no negative values
        for s in std:
            if s < 1:
                inverse_std.append(1 - s)
            else:
                inverse_std.append(0)

        inverse_std = np.array(inverse_std)

        # gets formula from dictionary
        factors = formulas[reweight_formula](inverse_std, mean)

        """
        # this statement assures that identities with images ("n") under a certain threshold remain unchanged
        if df_std.loc[i]["n"] <= 5:
            factors = np.zeros(40)
        """
        row_df = pd.DataFrame([factors], index=[i])

        # append to df
        df_factors = pd.concat([df_factors, row_df])

    return df_factors



"""
This function reweights attributes.
Make sure that the id corresponds with the indexes/ids from the other files.
If the parameter balanced is True, then the weights will be balanced based on the source and target distribution.
"""


def reweight_attributes(raw_file, df_mean, df_std, df_id, reweigh_formula):

    if raw_file == "list_attr_celeba.txt":
        df_raw = pd.read_csv("data/txt_files/list_attr_celeba.txt", sep='\s+', header=0)
        cols = df_raw.columns.tolist()
        cols = cols[1:] + [cols[0]]
        df_raw = df_raw[cols]
    else:
        df_raw = pd.read_csv(raw_file, header=0, index_col=0)

    df_raw = df_raw.merge(df_id, how="left", on="Image")

    # only keep pictures and their detected attributes of identities that are in the partition
    ids = np.unique(df_id.Id.values)

    # index is the id, rows 0-39 are the 40 attributes
    df_factors = get_weights(df_mean, df_std, ids, reweigh_formula).reset_index()

    # rename the columns
    columns = df_raw.columns.values.tolist()[:-2]
    columns.insert(0, "Id")
    df_factors.columns = columns

    # changing Id to int for easy comparison
    df_factors.Id = df_factors.Id.astype(int)

    df_corrected = apply_weights(ids, df_raw, df_factors)

    return df_corrected


def apply_weights(ids, df_raw, df_factors):
    df_corrected = pd.DataFrame()
    # apply weights to each picture of an identity
    for p, i in enumerate(ids):
        p = str(round(p / len(ids) * 100, 0))
        sys.stdout.write("\rReweighting file. Progress: " + p + "%")
        sys.stdout.flush()

        df_data = df_raw.loc[df_raw["Id"] == i]
        image = df_data.Image

        df_data = df_data.drop(columns=["Image", "Id"])
        # taking factors of the correct id
        attribute_factors = df_factors.loc[df_factors["Id"] == i]
        factors = np.array(attribute_factors.drop(columns="Id").values[0])
        # add to extraction values
        df_new = df_data.apply(lambda x: np.add(factors, x), raw=True, axis=1)
        df_new["Image"] = image
        df_corrected = df_corrected.append(df_new)
    print("\nDone!\n")
    return df_corrected
