import pandas as pd

from calculate_std_mean import calculate_statistics
from helper import get_ids_by_partition
from reweight_attributes import reweight_attributes
from calculate_accuracy import calculate_accuracy

testing_file = "extractions/new_testing_AFFACT1.txt"
# arcface ids:
df_id_af = pd.read_csv("ids/arcface_testing_ids.txt", sep='\s+', names=["Image", "Id"])
# ground truth:
df_id_gt = get_ids_by_partition(2)
df_std_gt, df_mean_gt = calculate_statistics(testing_file, df_id_gt, balanced=True)
df_std_af, df_mean_af = calculate_statistics(testing_file, df_id_af, balanced=True)

df_reweighed_af = reweight_attributes(testing_file, df_mean_af, df_std_af, df_id_af, reweigh_formula="square_sign")
df_reweighed_gt = reweight_attributes(testing_file, df_mean_gt, df_std_gt, df_id_gt, reweigh_formula="square_sign")



print("\nunweighed balanced unbalanced")
df_accuracy_og, df_detailed_og = calculate_accuracy(pd.read_csv(testing_file, header=0, index_col=0))
print("\nsquare_sign balanced unbalanced")
df_accuracy_af, df_detailed_af = calculate_accuracy(df_reweighed_af)
print("\nsquare_mean balanced unbalanced")
df_accuracy_gt, df_detailed_gt = calculate_accuracy(df_reweighed_gt)


"""
print("\ncube_sign balanced unbalanced")
df_accuracy3, df_detailed_3 = calculate_accuracy(df_reweighed3)
print("\ncube_mean balanced unbalanced")
df_accuracy4, df_detailed_4 = calculate_accuracy(df_reweighed4)
"""


# df_accuracy_og.to_csv("final_error_rates/AFFACT-A_og.csv")
df_accuracy_af.to_csv("error_rates/final/0_AFFACT-A_AF_square_sign_balanced.csv")
df_accuracy_gt.to_csv("error_rates/final/0_AFFACT-A_GT_square_sign_balanced.csv")


"""
df = pd.DataFrame(index=df_accuracy_af.index)
df["not reweighted"] = df_accuracy_og["Unbalanced"]
df["reweighted af"] = df_accuracy_af["Unbalanced"]
df["reweighted gt"] = df_accuracy_gt["Unbalanced"]
improvement_af = []
improvement_gt = []

for a in df_accuracy_af.index.to_list():
    before = df["not reweighted"][a]
    after_gt = df["reweighted gt"][a]
    after_af = df["reweighted af"][a]
    difference_gt = before - after_gt
    difference_af = before - after_af
    i_af = str(round((difference_af / before)*100, 2)) + "%"
    improvement_af.append(i_af)
    i_gt = str(round((difference_gt / before)*100, 2)) + "%"
    improvement_gt.append(i_gt)
df["change af"] = improvement_af
df["change gt"] = improvement_gt
df.to_csv("improvment_comparison_AFFACT2.csv")
"""


# df_accuracy1.to_csv("final_error_rates/AFFACT-B_AF_square_mean.csv")
# df_accuracy_og.to_csv("final_error_rates/AFFACT-A.csv")
# df_detailed_og.to_csv("final_error_rates/detailed/AFFACT-B_og.csv")
# df.to_csv("final_error_rates/AFFACT-B_GT_improvement_square_mean.csv")
# df_accuracy_og.to_csv("final_error_rates/AFFACT-B_no_reweight.csv")
# df_accuracy1.to_csv("final_error_rates/AFFACT-B_square_mean_unbalanced.csv")
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

