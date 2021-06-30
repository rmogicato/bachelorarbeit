import pandas as pd
import numpy as np

from helper import calculate_distribution

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

# df_reweighed = pd.read_csv("extractions/new_testing_AFFACT1.txt", header=0, index_col=0)

# df = df_reweighed

# df = df.sort_values(by="Image")


# binary sorting
def binarize(x):
    if isinstance(x, float):
        if x < 0:
            x = -1
        else:
            x = 1
    return x


def calculate_accuracy(df):
    df = df.sort_values(by="Image")
    images = df["Image"].values

    df_automatic = df.applymap(binarize).set_index(df["Image"]).drop(columns="Image")

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

    df_calculated_results = pd.DataFrame(columns=["Balanced equal", "Balanced adapted", "Unbalanced"], index=columns)

    for key in results:
        row = results[key]

        sensitivity = row["correct_positives"] / (row["correct_positives"] + row["false_negatives"])
        specificity = row["correct_negatives"] / (row["false_positives"] + row["correct_negatives"])
        accuracy = (sensitivity + specificity) / 2
        balanced_accuracy.append(accuracy)
        df_calculated_results["Balanced equal"][key] = 1 - accuracy

    average_balanced = 1 - np.array(balanced_accuracy).mean()
    print(round((1 - average_balanced)*100,2))

    unbalanced_accuracy = []
    for key in results:
        row = results[key]
        accuracy = (row["correct_positives"] + row["correct_negatives"]) / (
                    row["correct_positives"] + row["correct_negatives"] + row["false_positives"] + row[
                "false_negatives"])
        unbalanced_accuracy.append(1 - accuracy)
        df_calculated_results["Unbalanced"][key] = 1 - accuracy
        # print(key, accuracy)
    average_unbalanced = np.array(unbalanced_accuracy).mean()
    print(round((1 - average_unbalanced)*100, 2))
    df_target_dist = pd.read_csv("training_attribute_distribution.csv", index_col=0, header=0)

    balanced_er = []
    for key in results:
        row = results[key]
        s1 = row["false_negatives"]*df_target_dist["positive"][key] / (row["correct_positives"]+row["false_negatives"])
        s2 = row["false_positives"]*df_target_dist["negative"][key] / (row["correct_negatives"]+row["false_positives"])
        ber = s1 + s2
        balanced_er.append(ber)
        df_calculated_results["Balanced adapted"][key] = ber
    average_balanced2 = np.array(balanced_er).mean()
    print("average: ", average_balanced2)
    # df_calculated_results["Unbalanced"][key] = 1 - accuracy
    """
    balanced_er2 = []
    for key in results:
        row = results[key]
        errors = 0
        df_errors = df_compared[key].loc[(df_compared[key]["self"] != df_compared[key]["other"])]
        for image in df_errors.index.tolist():
            if df_errors["other"][image] == 1:
                e = df_target_dist["positive"][key] / (row["correct_positives"]+row["false_negatives"])
                errors += e
            else:
                e = df_target_dist["negative"][key] / (row["correct_negatives"]+row["false_positives"])
                errors += e
        balanced_er2.append(errors)
    print("average 2: ", np.array(balanced_er2).mean())
    """
    df_calculated_results.loc["Average"] = [average_balanced, average_balanced2, average_unbalanced]
    return df_calculated_results




