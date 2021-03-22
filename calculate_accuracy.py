import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

df_testing = pd.read_csv("extracted_attributes_testing.txt", header=0, index_col=0)
df_validation = pd.read_csv("extracted_attributes_validation.txt", header=0, index_col=0)
df_training = pd.read_csv("extracted_attributes_training.txt", header=0, index_col=0)
df = pd.concat([df_testing, df_training, df_validation])

df = df.sort_values(by="Image")
df = df.reset_index().drop(columns=["index"])


# binary sorting
def foo_bar(x):
    if isinstance(x, float):
        if x < 0:
            x = -1
        else:
            x = 1
    return x


df_automatic = df.applymap(foo_bar).drop(columns="Image")

df_manual = pd.read_csv("list_attr_celeba.txt", sep='\s+', header=0).drop(columns="Image")

df_compared = df_automatic.compare(df_manual, keep_shape=True)

# number of NaN in compared data frame i.d. number of values that are the same
# we divide this by two, as each entry is counted twice
nans = df_compared.isna().sum()
columns = df_manual.columns
rows = df_manual.shape[0]
total_correct = 0

for col in columns:
    correct = nans[col].self
    total_correct += correct
    print(col, round((1 - correct / rows) * 100, 2))

print("total accuracy: ", (1 - total_correct / (rows * 40)) * 100)
