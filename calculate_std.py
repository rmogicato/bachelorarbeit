import pandas as pd
import sys
from sklearn import preprocessing

pd.set_option('display.max_columns', None)


# gets data frame from txt files
# the boolean automatic determines whether it returns data from the automatic extracted attributes or from the manual ones
def get_df(automatic):
    df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
    if automatic:
        df = pd.read_csv("data-AFFACT2/extracted_attributes_validation_AFFACT2.txt", header=0, index_col=0)
        # sort dataframe by image
        df = df.sort_values(by="Image")
        # reset index and drop index column
        df = df.reset_index(drop="index")

        # merge on Image to get Id for each image
        df = df.merge(df_id, how="inner", on="Image")
        # todo: look whether normalization is worthwhile
    else:
        df = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0)
        df["Id"] = df_id["Id"]
        df = df.sort_values(by="Id")

    return df


df_attr = get_df(True)
print(df_attr)

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
    sys.stdout.write("\rProgress: " + p + "%")
    sys.stdout.flush()

df_std.to_csv("data-AFFACT2/std_automatic_validation.calculated_csv")
df_mean.to_csv("data-AFFACT2/mean_automatic_validation.calculated_csv")

