import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

df_reweighed = pd.read_csv("extractions/reweighed/gt_testing_reweighed_square_mean_AFFACT2.txt", header=0, index_col=0)

df = df_reweighed

df = df.sort_values(by="Image")


# binary sorting
def foo_bar(x):
    if isinstance(x, float):
        if x < 0:
            x = -1
        else:
            x = 1
    return x


images = df["Image"].values

df_automatic = df.applymap(foo_bar).set_index(df["Image"]).drop(columns="Image")

# ground truth of all identities
df_raw = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
df_manual = df_raw.loc[df_raw["Image"].isin(images)]
df_manual = df_manual.set_index(df_manual["Image"]).drop(columns="Image")
print(df_manual)
print(df_automatic)

df_compared = df_automatic.compare(df_manual, keep_shape=True, keep_equal=True)

print(df_compared)


columns = df_manual.columns
rows = df_manual.shape[0]
false_positive = 0
false_negatives = 0
results = {}

# iterating over each attribute

for col in columns:
    false_positives = len(df_compared[col].loc[(df_compared[col]["self"] == 1.0) & (df_compared[col]["other"] == -1.0)].values)
    false_negatives = len(df_compared[col].loc[(df_compared[col]["self"] == -1.0) & (df_compared[col]["other"] == 1.0)].values)
    correct_positives = len(df_compared[col].loc[(df_compared[col]["self"] == 1.0) & (df_compared[col]["other"] == 1.0)].values)
    correct_negatives = len(df_compared[col].loc[(df_compared[col]["self"] == -1.0) & (df_compared[col]["other"] == -1.0)].values)
    results[col] = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "correct_positives":  correct_positives,
        "correct_negatives": correct_negatives,
    }

df_results = pd.DataFrame(data=results)
df_results.to_csv("error_rates/er_gt_square_mean_AFFACT2")
print(results)
balanced_accuracy = []

for key in results:
    row = results[key]

    sensitivity = row["correct_positives"] / (row["correct_positives"]+row["false_negatives"])
    specificity = row["correct_negatives"] / (row["false_positives"]+row["correct_negatives"])
    accuracy = (sensitivity + specificity)/2
    balanced_accuracy.append(accuracy)
    # print(key, accuracy)

average_balanced = np.array(balanced_accuracy).mean()
print(1-average_balanced)


unbalanced_accuracy = []
for key in results:
    row = results[key]
    accuracy = (row["correct_positives"]+row["correct_negatives"])/(row["correct_positives"]+row["correct_negatives"]+row["false_positives"]+row["false_negatives"])
    unbalanced_accuracy.append(accuracy)
    # print(key, accuracy)
average_unbalanced = np.array(unbalanced_accuracy).mean()
print(1-average_unbalanced)
