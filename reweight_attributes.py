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

        row_df = pd.DataFrame([factors], index=[i])

        # append to df
        df_factors = pd.concat([df_factors, row_df])

    return df_factors


def balance_weighs(df_factors, df_source_dist, df_raw):
    # calculating target distribution from df_raw
    df_target_dist = calculate_distribution(df_raw)
    # calculating probability based on target and source distribution
    df_probability = calculate_probability(df_source_dist)
    df_factors_new = df_factors
    # balancing all the weights
    for a in df_probability.index.tolist():
        factors = df_factors[a].values
        reweighed_factors = []
        for f in factors:
            # if factor is negative with the negative probability
            if f < 0:
                p = df_probability["negative"][a]
                reweighed_factors.append(f * p)
            # else with the positive factor
            else:
                p = df_probability["positive"][a]
                reweighed_factors.append(f * p)
        df_factors_new[a] = reweighed_factors
    return df_factors_new


"""
this function computes the probability associated with each cluster.
The formula was taken from MOON, arXiv:1603.07027
"""


def calculate_probability(df_source_dist):
    attributes = df_source_dist.index.tolist()
    df_probability = pd.DataFrame(columns=["positive", "negative"], index=attributes)
    for a in attributes:
        t_p = 0.5
        s_p = df_source_dist["positive"][a]
        t_n = 0.5
        s_n = df_source_dist["negative"][a]
        if t_p > s_p:
            p_positive = 1
        else:
            p_positive = (s_n * t_p) / (s_p * t_n)
        if t_n > s_n:
            p_negative = 1
        else:
            p_negative = (s_p * t_n) / (s_n * t_p)
        df_probability["positive"][a] = p_positive
        df_probability["negative"][a] = p_negative
    return df_probability


"""
This function reweights attributes.
Make sure that the id corresponds with the indexes/ids from the other files.
If the parameter balanced is True, then the weights will be balanced based on the source and target distribution.
"""


def reweight_attributes(raw_file, df_mean, df_std, df_id, reweigh_formula, balanced):

    if raw_file == "list_attr_celeba.txt":
        df_raw = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
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

    if balanced:
        df_source_dis = pd.read_csv("training_attribute_distribution.csv", index_col=0, header=0)
        df_factors = balance_weighs(df_factors, df_source_dis, df_raw)

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


"""
raw_file = "extractions/new_testing_AFFACT1.txt"
mean_file = "extractions/means/mean_arcface_testing_AFFACT1.txt"
std_file = "extractions/stds/std_arcface_testing_AFFACT1.txt"
df_id = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Id"])

df_mean = pd.read_csv(mean_file, header=0, index_col=0).reset_index()
df_std = pd.read_csv(std_file, header=0, index_col=0).reset_index()

df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
partition = pd.read_csv("list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
images = partition.loc[partition.Partition == 2].Image.values
df_id = df_id.loc[df_id.Image.isin(images)]

corrected = reweight_attributes(raw_file, df_mean, df_std, df_id, reweigh_formula="square_mean", balanced=True)
corrected = corrected.sort_values(by="Image")

print(corrected)
# corrected.to_csv("extractions/reweighed/gt_testing_reweighed_cube_mean_AFFACT1.txt")
"""
