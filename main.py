import pandas as pd
import numpy as np

from calculate_std import calculate_statistics
from reweight_attributes import reweight_attributes
from calculate_accuracy import calculate_accuracy


def get_ids_by_partition(partition):
    df_ids = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
    df_partition = pd.read_csv("list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
    df_partition = df_partition.loc[df_partition.Partition == partition]
    df_ids = df_ids.merge(df_partition, on="Image").drop(columns="Partition")
    return df_ids


testing_file = "extractions/new_testing_AFFACT1.txt"
# arcface ids:
# df_id = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Id"])
# ground truth:
df_id = get_ids_by_partition(2)

df_std, df_mean = calculate_statistics(testing_file, df_id)
df_reweighed1 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="square_sign", balanced=True)
"""
df_reweighed2 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="square_mean", balanced=True)
df_reweighed3 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="cube_sign", balanced=True)
df_reweighed4 = reweight_attributes(testing_file, df_mean, df_std, df_id, reweigh_formula="cube_mean", balanced=True)
"""

print("\nunweighed balanced unbalanced")
df_accuracy_og = calculate_accuracy(pd.read_csv(testing_file, header=0, index_col=0))
print("\nsquare_sign balanced unbalanced")
df_accuracy1 = calculate_accuracy(df_reweighed1)

print(df_accuracy_og)
print(df_accuracy1)
"""
print("\nsquare_mean balanced unbalanced")
df_accuracy2 = calculate_accuracy(df_reweighed2)
print("\ncube_sign balanced unbalanced")
df_accuracy3 = calculate_accuracy(df_reweighed3)
print("\ncube_mean balanced unbalanced")
df_accuracy4 = calculate_accuracy(df_reweighed4)
"""
"""
df_accuracy1.to_csv("error_rates/final/gt_AFFACT2_testing_unbalanced_square_sign.csv")
df_accuracy2.to_csv("error_rates/final/gt_AFFACT2_testing_unbalanced_square_mean.csv")
df_accuracy3.to_csv("error_rates/final/gt_AFFACT2_testing_unbalanced_cube_sign.csv")
df_accuracy4.to_csv("error_rates/final/gt_AFFACT2_testing_unbalanced_cube_mean.csv")
df_accuracy_og.to_csv("error_rates/final/gt_AFFACT2_testing_no_reweight.csv")
"""

