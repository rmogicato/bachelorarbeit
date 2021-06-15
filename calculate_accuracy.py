import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

df_reweighed = pd.read_csv("extractions/new_testing_AFFACT1.txt", header=0, index_col=0)

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


def calculate_accuracy(df):
    df = df.sort_values(by="Image")
    images = df["Image"].values

    df_automatic = df.applymap(foo_bar).set_index(df["Image"]).drop(columns="Image")

    # ground truth of all identities
    df_raw = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
    df_manual = df_raw.loc[df_raw["Image"].isin(images)]
    df_manual = df_manual.set_index(df_manual["Image"]).drop(columns="Image")

    df_compared = df_automatic.compare(df_manual, keep_shape=True, keep_equal=True)

    columns = df_manual.columns
    results = {}

    # iterating over each attribute

    for col in columns:
        false_positives = len(
            df_compared[col].loc[(df_compared[col]["self"] == 1.0) & (df_compared[col]["other"] == -1.0)].values)
        false_negatives = len(
            df_compared[col].loc[(df_compared[col]["self"] == -1.0) & (df_compared[col]["other"] == 1.0)].values)
        correct_positives = len(
            df_compared[col].loc[(df_compared[col]["self"] == 1.0) & (df_compared[col]["other"] == 1.0)].values)
        correct_negatives = len(
            df_compared[col].loc[(df_compared[col]["self"] == -1.0) & (df_compared[col]["other"] == -1.0)].values)
        results[col] = {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "correct_positives": correct_positives,
            "correct_negatives": correct_negatives,
        }

    df_results = pd.DataFrame(data=results)
    balanced_accuracy = []

    df_calculated_results = pd.DataFrame(columns=["Balanced", "Unbalanced"], index=columns)

    for key in results:
        row = results[key]

        sensitivity = row["correct_positives"] / (row["correct_positives"] + row["false_negatives"])
        specificity = row["correct_negatives"] / (row["false_positives"] + row["correct_negatives"])
        accuracy = (sensitivity + specificity) / 2
        balanced_accuracy.append(accuracy)
        df_calculated_results["Balanced"][key] = 1 - accuracy

    average_balanced = np.array(balanced_accuracy).mean()
    print("Average balanced error rate: ", 1 - average_balanced)

    unbalanced_accuracy = []
    for key in results:
        row = results[key]
        accuracy = (row["correct_positives"] + row["correct_negatives"]) / (
                    row["correct_positives"] + row["correct_negatives"] + row["false_positives"] + row[
                "false_negatives"])
        unbalanced_accuracy.append(accuracy)
        df_calculated_results["Unbalanced"][key] = 1 - accuracy
        # print(key, accuracy)
    average_unbalanced = np.array(unbalanced_accuracy).mean()
    print("Average unbalanced error rate: ", 1 - average_unbalanced)

    return df_results
