import pandas as pd

from calculate_std_mean import calculate_statistics
from helper import get_ids_by_partition
from reweight_attributes import reweight_attributes
from calculate_accuracy import calculate_accuracy

"""
This file is an example how the entire reweighting process works.
As an input file we use extractions of AFFACT-B.
This is a balanced neural network, this means we calculate
"""
# setting file
testing_file = "extractions/new_testing_AFFACT1.txt"
# arcface ids:
df_id_af = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Id"])
# ground truth ids:
df_id_gt = get_ids_by_partition(2)

# calculating mean and std with both labels
df_std_gt, df_mean_gt = calculate_statistics(testing_file, df_id_gt, balanced=True)
df_std_af, df_mean_af = calculate_statistics(testing_file, df_id_af, balanced=True)

# reweighting with both labels
df_reweighed_af = reweight_attributes(testing_file, df_mean_af, df_std_af, df_id_af, reweigh_formula="square_sign")
df_reweighed_gt = reweight_attributes(testing_file, df_mean_gt, df_std_gt, df_id_gt, reweigh_formula="square_sign")



print("\nunweighed (balanced, unbalanced)")
df_accuracy_og, df_detailed_og = calculate_accuracy(pd.read_csv(testing_file, header=0, index_col=0))
print("\nsquare_sign arcface ids(balanced, unbalanced)")
df_accuracy_af, df_detailed_af = calculate_accuracy(df_reweighed_af)
print("\nsquare_mean ground truth ids (balanced, unbalanced)")
df_accuracy_gt, df_detailed_gt = calculate_accuracy(df_reweighed_gt)


df_accuracy_og.to_csv("final_error_rates/AFFACT-A_og.csv")
df_accuracy_af.to_csv("error_rates/final/0_AFFACT-A_AF_square_sign_balanced.csv")
df_accuracy_gt.to_csv("error_rates/final/0_AFFACT-A_GT_square_sign_balanced.csv")
