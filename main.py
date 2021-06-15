import pandas as pd
import numpy as np

from calculate_std import calculate_statistics
from reweight_attributes import reweight_attributes
from calculate_accuracy import calculate_accuracy

testing_file = "extractions/new_testing_AFFACT1.txt"

df_id = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Id"])

df_std, df_mean = calculate_statistics(testing_file, df_id)
df_reweighed1 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="square_sign", balanced=True)
df_reweighed2 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="square_mean", balanced=True)
df_reweighed3 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="cube_sign", balanced=True)
df_reweighed4 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="cube_mean", balanced=True)

print("\nsquare_sign ")
df_accuracy1 = calculate_accuracy(df_reweighed1)
print("\nsquare_mean")
df_accuracy2 = calculate_accuracy(df_reweighed2)
print("\ncube_sign ")
df_accuracy3 = calculate_accuracy(df_reweighed3)
print("\ncube_mean")
df_accuracy4 = calculate_accuracy(df_reweighed4)

print("\nunweighed")
df_accuracy_og = calculate_accuracy(pd.read_csv(testing_file, header=0, index_col=0))

# df_accuracy1.to_csv("AFFACT_balanced/er_AFFACT1_balanced_cube_mean_.csv")
# df_accuracy2.to_csv("AFFACT_balanced/er_AFFACT1_unbalanced_cube_sign.csv")
# df_accuracy3.to_csv("AFFACT_balanced/er_AFFACT1_validation_og.csv")
