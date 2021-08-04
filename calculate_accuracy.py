import pandas as pd
import numpy as np

from helper import calculate_distribution

# this function returns -1 for all values smaller than 0, and 1 for all other values
def binarize(x):
    if isinstance(x, float):
        if x < 0:
            x = -1
        else:
            x = 1
    return x

"""
This function calculates the accuracy of extracted values (both reweighted and not reweighted)
It returns two dataframes, a undetailed one and a detailed one.
Former is suited for easy comparison and  simply contains the balanced and unbalanced error rates in percent,
rounded to two decimal places.
Latter additionally contains the false negative rate and false positive rate. Here the values are not rounded and
not in percent.
"""


def calculate_accuracy(df):
    df = df.sort_values(by="Image")
    images = df["Image"].values

    df_automatic = df.applymap(binarize).set_index(df["Image"]).drop(columns="Image")

    # ground truth of all identities
    df_raw = pd.read_csv("data/txt_files/list_attr_celeba.txt", sep='\s+', header=0)
    df_manual = df_raw.loc[df_raw["Image"].isin(images)]
    df_manual = df_manual.set_index(df_manual["Image"]).drop(columns="Image")

    # comparing both data frames to see the differences, the columns "self" show the extracted values,
    # "other" the ground truth values
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

    error_rate_types = ["false_negative_rate", "false_positive_rate", "balanced_error_rate", "unbalanced_error_rate"]
    df_detailed = pd.DataFrame(index=columns, columns=error_rate_types)
    balanced_accuracy = []

    df_calculated_results = pd.DataFrame(columns=["Balanced", "Unbalanced"], index=columns)

    # iterating over each attribute to obtain the balanced error rate
    for key in results:
        row = results[key]

        false_negative_rate = row["false_negatives"] / (row["false_negatives"] + row["correct_positives"] )
        false_positive_rate = row["false_positives"] / (row["false_positives"] + row["correct_negatives"])
        error_rate = (false_negative_rate + false_positive_rate)/2
        balanced_accuracy.append(error_rate)
        df_detailed["false_negative_rate"][key] = false_negative_rate
        df_detailed["false_positive_rate"][key] = false_positive_rate
        df_detailed["balanced_error_rate"][key] = error_rate
        df_calculated_results["Balanced"][key] = round(error_rate*100, 2)

    average_balanced = round(np.array(balanced_accuracy).mean()*100, 2)

    # iterating over each attribute to obtain the accuracy
    unbalanced_accuracy = []
    for key in results:
        row = results[key]
        accuracy = (row["correct_positives"] + row["correct_negatives"]) / (
                    row["correct_positives"] + row["correct_negatives"] + row["false_positives"] + row[
                "false_negatives"])
        unbalanced_accuracy.append(1 - accuracy)
        df_calculated_results["Unbalanced"][key] = round((1 - accuracy)*100, 2)
        df_detailed["unbalanced_error_rate"][key] = 1 - accuracy
    average_unbalanced = round(np.array(unbalanced_accuracy).mean()*100, 2)

    df_calculated_results.loc["Average"] = [average_balanced, average_unbalanced]

    # printing both values
    print(average_balanced, average_unbalanced)
    averages = []
    # calculate averages for each error rate type
    for t in error_rate_types:
        column = np.array(df_detailed[t].values)
        averages.append(column.mean())
    df_detailed.loc["Average"] = averages
    return df_calculated_results, df_detailed




