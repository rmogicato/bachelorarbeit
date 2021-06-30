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