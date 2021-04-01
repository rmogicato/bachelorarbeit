import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

df_validation = pd.read_csv("data-AFFACT2/extracted_attributes_validation_corrected.txt", header=0, index_col=0)
df = df_validation

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

df_raw = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
df_manual = df_raw.loc[df_raw["Image"].isin(images)]
df_manual = df_manual.set_index(df_manual["Image"]).drop(columns="Image")
print(df_manual)
print(df_automatic)

df_compared = df_automatic.compare(df_manual, keep_shape=True)

print(df_compared)

# number of NaN in compared data frame i.d. number of values that are the same
# we divide this by two, as each entry is counted twice
columns = df_manual.columns
rows = df_manual.shape[0]
false_positive = 0
false_negatives = 0
results = {}
# iterating over each attribute

for col in columns:
    type1 = len(df_compared[col].loc[df_compared[col]["self"] == 1.0].values)
    type2 = len(df_compared[col].loc[df_compared[col]["self"] == -1.0].values)
    correct = rows - type1 - type2
    fp_rate = round((type1 / rows) * 100, 2)
    fn_rate = round((type2 / rows) * 100, 2)
    results[col] = {
        "false_positive": type1,
        "false_negatives": type2,
        "correct": correct,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "error_rate": fp_rate + fn_rate
    }

    # total_correct += correct
    # print(col, round((1 - correct / rows) * 100, 2))

# print("total accuracy: ", (1 - total_correct / (rows * 40)) * 100)
df_results = pd.DataFrame(data=results)
total = np.array(df_results.T.error_rate.values)

print(df_results.T.error_rate)
print(np.mean(total))

df_results.to_csv("error_rates/extracted_attributes_validation_corrected_AFFACT2")
