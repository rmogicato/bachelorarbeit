from reweight_attributes import balance_weighs, calculate_probability
from helper import calculate_distribution
import pandas as pd

df_raw = pd.read_csv("../extractions/new_testing_AFFACT1.txt", header=0, index_col=0)

df_target_dist = calculate_distribution(df_raw)
df_source_dist = pd.read_csv("../training_attribute_distribution.csv", header=0, index_col=0)
df_probability = calculate_probability(df_source_dist=df_source_dist)
df_probability = df_probability.astype(float).round(2)
df_probability.to_csv("probability_testing_AFFACT1.csv")
