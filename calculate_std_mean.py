import pandas as pd
import numpy as np
import sys
import os

# gets data frame from txt files
from helper import calculate_probability, calculate_distribution, get_ids_by_partition


# getting data frame
def get_df(filename, df_id):
    if filename == "data/txt_files/list_attr_celeba.txt":
        dir_path = os.path.dirname(os.path.realpath(__file__))

        df = pd.read_csv(dir_path + "/" + filename, sep='\s+', header=0)
        df = df.merge(df_id, on="Image")
        df = df.sort_values(by="Id")
    else:
        df = pd.read_csv(filename, header=0, index_col=0)
        # sort dataframe by image
        df = df.sort_values(by="Image")
        # reset index and drop index column
        df = df.reset_index(drop="index")

        # merge on Image to get Id for each image
        df = df.merge(df_id, how="inner", on="Image")
    return df


"""
This function calculates the mean and standard deviation of a file.
Provide the id with a df_id that contains the image and id (ground truth/clustered) on which the calculations should be based on.
"""


def calculate_statistics(filename, df_id, balanced=False):

    df_attr = get_df(filename, df_id)

    # creating df for standard deviation for each person, dropping the column for image
    columns_std = df_attr.columns.to_list()
    columns_std.remove("Image")  # dropping image
    columns_std.remove("Id")  # dropping id
    columns_std.append("n")

    df_std = pd.DataFrame(columns=columns_std)
    df_mean = pd.DataFrame(columns=columns_std)

    # reading out ids and sorting the list
    ids = df_attr["Id"].drop_duplicates().to_list()
    ids = sorted(ids)

    # calculating the probability which we later use if we use the balanced std/mean
    training_ids = get_ids_by_partition(0)

    cols = df_attr.columns.tolist()
    cols.remove("Image")
    cols.remove("Id")
    training_df = get_df("data/txt_files/list_attr_celeba.txt", training_ids)

    # making sure that the columns are in the same order
    training_df = training_df[cols]
    df_source_dist = calculate_distribution(training_df)
    df_probability = calculate_probability(df_source_dist)
    df_probability = df_probability.astype(float).round(2)  # rounding to two decimal places

    for i, identity in enumerate(ids):
        # take all rows with the correct id (all of the same person)
        df_person = df_attr.loc[df_attr["Id"] == identity]

        # if balanced, we modify values according to probability
        if balanced:
            means = {}

            # iterating through each attribute
            for j, a in enumerate(df_person.columns.to_list()[:40]):
                values = df_person[a].values
                balanced_values = []
                # modifying each value according to the probability of that attribute
                for v in values:
                    # negative values are modified with the negative probability
                    if v < 0:
                        p = df_probability.loc[a]["negative"]
                        balanced_values.append(v * p)
                    # positive values are modified with the positive probability
                    else:
                        p = df_probability.loc[a]["positive"]
                        balanced_values.append(v * p)
                means[a] = np.sum(balanced_values) / len(df_person.index)
            means["n"] = len(df_person.index)
            df_mean.loc[identity] = means

            stds = {}
            # now we calculate the standard deviation of each attribute using the modified values
            for j, a in enumerate(df_person.columns.to_list()[:40]):
                xs = df_person[a].values
                # difference between each value and the mean of a of i
                dif = (xs - means[a])**2
                # variance
                var = np.sum(dif) / len(df_person.index)
                std = np.sqrt(var)
                stds[a] = std
            stds["n"] = len(df_person.index)
            df_std.loc[identity] = stds

        # otherwise we just calculate the normal mean and standard deviation
        else:
            # adds values to df of std
            std = df_person.std(axis=0, ddof=0)
            std["n"] = len(df_person.index)
            df_std.loc[identity] = std

            # adds values to df of mean
            mean = df_person.mean(axis=0)
            mean["n"] = len(df_person.index)
            df_mean.loc[identity] = mean

        # printing progress
        p = str(round(i / len(ids) * 100, 0))
        sys.stdout.write("\r Calculating standard deviation and mean. Progress: " + p + "%")
        sys.stdout.flush()
    print("\nDone!\n")
    return df_std, df_mean
