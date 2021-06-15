import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_attr = pd.read_csv("list_attr_celeba.txt", sep='\s+',)
attributes = df_attr.columns.to_list()[1:]


def get_distribution(df, attributes):
    p = []
    n = []
    for a in attributes:
        col = np.array(df[a].to_list())
        positive = np.count_nonzero(col == 1) / len(col)
        negative = 1 - positive
        p.append(positive)
        n.append(negative)
    return p, n


positives, negatives = get_distribution(df_attr, attributes)

fig, ax = plt.subplots()

plt.xticks(rotation=90)
ax.bar(attributes, positives, 0.6, label='Positive')
ax.bar(attributes, negatives,  0.6, bottom=positives, label='Negative')
ax.set_ylabel('Ratio')
ax.set_title('Class imbalance')
ax.legend()

plt.show()

"""
Below we generate a histogram that shows how many pictures are associated per identities

"""


df_id = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
df_partition = pd.read_csv("list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
df_id = pd.concat([df_id, df_partition.Partition], axis=1, join="outer")

df_id = df_id.loc[df_id["Partition"].isin([0,1,2])]

img_per_id = df_id["Id"].value_counts()
median = np.median(np.array(img_per_id).mean())
print("median", median)

array = np.array(img_per_id.to_list())

# this threshold prints how many identities have fewer pictures than the threshold
threshold = 5
under_threshold = np.count_nonzero(array <= threshold)
print("under threshold: ", under_threshold)

# generate histogram with a number of bins
h = plt.hist(img_per_id, bins=35)
plt.show()

"""
This code calculates the source distribution of the training partition
"""

df_distribution = pd.DataFrame(index=attributes, columns=["positive", "negative"])
df_training = df_attr.merge(df_partition, on="Image")
# selecting training partition
df_training = df_training.loc[df_training["Partition"] == 0]
df_training = df_training.drop(columns=["Partition"])
df_distribution["positive"], df_distribution["negative"] = get_distribution(df_training, attributes)
df_distribution.to_csv("training_attribute_distribution.csv")