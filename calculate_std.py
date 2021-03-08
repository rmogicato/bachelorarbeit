import pandas as pd
import sys


# gets data frame from txt files
def get_df():
    df = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
    pd.set_option('display.max_columns', None)
    df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])

    df["Id"] = df_id["Id"]
    df = df.sort_values(by="Id")
    # df_attr = df_attr.head(1000)
    return df


df_attr = get_df()

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

for i in ids:
    # take all rows with the correct id (all of the same person)
    df_person = df_attr.loc[df_attr["Id"] == i]

    # adds values to df of std
    std = df_person.std(axis=0, ddof=0)
    std["n"] = len(df_person.index)
    df_std.loc[i] = std

    # adds values to df of mean
    mean = df_person.mean(axis=0)
    mean["n"] = len(df_person.index)
    df_mean.loc[i] = mean

    # printing progress
    p = str(round(i / len(ids) * 100, 0))
    sys.stdout.write("\rProgress: " + p + "%")
    sys.stdout.flush()

df_std.to_csv("std.csv")
df_mean.to_csv("mean.csv")
