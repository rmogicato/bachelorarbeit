import numpy as np
import pandas as pd


def calculate_distribution(df_raw):
    attributes = df_raw.columns[:40].tolist()
    df_dist = pd.DataFrame(index=attributes, columns=["positive", "negative"])
    for a in attributes:
        col = np.array(df_raw[a].to_list())
        positive = np.count_nonzero(col > 0) / len(col)
        negative = 1 - positive
        df_dist["positive"][a] = positive
        df_dist["negative"][a] = negative
    return df_dist


"""
This function computes the probability according to a source and target domain.
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
