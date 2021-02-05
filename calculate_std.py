import sys

import pandas as pd

df_attr = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
pd.set_option('display.max_columns', None)
df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])

df_attr["Id"] = df_id["Id"]
df_attr = df_attr.sort_values(by="Id")

# creating df for standard deviation for each person, dropping the column for image
columns_std = df_attr.columns.to_list()
columns_std.remove("Image") #dropping image
columns_std.remove("Id") #dropping id
columns_std.append("Number")

df_std = pd.DataFrame(columns=columns_std)

# reading out ids and sorting the list
ids = df_attr["Id"].drop_duplicates().to_list()
ids = sorted(ids)

for i in ids:
    # take all rows with the correct id
    df_person = df_attr.loc[df_attr["Id"] == i]
    std = df_person.std(axis=0, ddof=0)
    std["Number"] = len(df_person.index)
    df_std.loc[i] = std

    progress = round(i / len(ids) * 100, 2)
    text = "\rProgress:  " + str(progress) + "%"
    sys.stdout.write(text)

df_std.to_csv(path_or_buf="./std.csv")
