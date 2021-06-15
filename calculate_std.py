import pandas as pd
import sys
from sklearn import preprocessing


# gets data frame from txt files
def get_df(filename, df_id):
    if filename == "list_attr_celeba.txt":
        df = pd.read_csv(filename, sep='\s+', header=0)
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


def calculate_statistics(filename, df_id):

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

    for i, identity in enumerate(ids):
        # take all rows with the correct id (all of the same person)
        df_person = df_attr.loc[df_attr["Id"] == identity]

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

"""
df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
partition = pd.read_csv("list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
images = partition.loc[partition.Partition == 2].Image.values
df_id = df_id.loc[df_id.Image.isin(images)]


df_std, df_mean = calculate_statistics("list_attr_celeba.txt", df_id)

# df_mean.to_csv("extractions/means/mean_gt_testing_AFFACT1.txt")
# df_std.to_csv("extractions/stds/std_gt_testing_AFFACT1.txt")
"""