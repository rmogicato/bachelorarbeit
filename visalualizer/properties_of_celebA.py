import bokeh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import show
from matplotlib.pyplot import figure
from matplotlib import rc
plt.rcParams['font.family'] = 'Palatino Linotype'

"""
This file explored the various properties of celebA.
"""


df_attr = pd.read_csv("../data/txt_files/list_attr_celeba.txt", sep='\s+', )
attributes = df_attr.columns.to_list()[1:]
attributes_clean = []

for a in attributes:
    attributes_clean.append(a.replace("_", " "))
print(attributes_clean)


# counting negative and positive instances
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

f = figure(figsize=(8, 4), dpi=80)

bars = np.add(positives, negatives).tolist()
print(bars)
r = range(40)
barWidth = 1

plt.title("Class Distribution")
plt.bar(r, positives, color="#c9d9d3", edgecolor='black', width=barWidth)
plt.bar(r, negatives, color="#718dbf", bottom=positives, edgecolor='black', width=barWidth, hatch="x")
plt.xticks(r, attributes_clean, rotation='vertical')
plt.margins(x=0, y=0)

plt.show()
f.savefig("class_ratio.pdf")

"""
Below we generate a histogram that shows how many pictures are associated per identities
"""


df_id = pd.read_csv("../data/txt_files/identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
df_partition = pd.read_csv("../data/txt_files/list_eval_partition.txt", sep='\s+', names=["Image", "Partition"])
df_id = pd.concat([df_id, df_partition.Partition], axis=1, join="outer")

# choose which partitions should be considered
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

