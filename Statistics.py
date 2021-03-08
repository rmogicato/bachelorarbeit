import pandas as pd
import sys

pd.set_option('display.max_columns', None)

# loading provided attributes and saving them in data frame df_attr
df_attr = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)


df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])

# adding a column Id to each
df_attr["Id"] = df_id["Id"]
df_attr = df_attr.sort_values(by="Id")

df_attr_automatic = pd.read_csv("extracted_attributes.csv", header=0)


# df_attr = df_attr.head(1000)
print(df_attr)

# creating df for standard deviation for each person, dropping the column for image
columns_std = df_attr.columns.to_list()
columns_std.remove("Image") #dropping image
columns_std.remove("Id") #dropping id
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
    p = str(round(i/len(ids)*100, 0))
    sys.stdout.write("\rProgress: " + p + "%")
    sys.stdout.flush()

df_std.to_csv("std.csv")
df_mean.to_csv("mean.csv")

