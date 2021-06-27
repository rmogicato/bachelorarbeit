import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import show
from bokeh.plotting import figure

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
print(positives)
print(negatives)

classes = ["positive", "negative"]
colors = ["#c9d9d3", "#718dbf"]

data = {'attributes': attributes,
        'positive': positives,
        'negative': negatives
        }

p = figure(x_range=attributes, plot_height=350, title="Attribute distribution",
           toolbar_location=None)

p.vbar_stack(classes, x='attributes', width=1, color=colors, source=data,
             legend_label=classes, hatch_pattern=[" ", "x"], hatch_weight=1, line_color="black", line_width=1)

p.y_range.start = 0
p.x_range.range_padding = 0
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.xaxis.major_label_orientation = "vertical"

show(p)

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

# todo: add to helper file
df_distribution = pd.DataFrame(index=attributes, columns=["positive", "negative"])
df_training = df_attr.merge(df_partition, on="Image")
# selecting training partition
df_training = df_training.loc[df_training["Partition"] == 2]
df_training = df_training.drop(columns=["Partition"])
df_distribution["positive"], df_distribution["negative"] = get_distribution(df_training, attributes)
df_distribution.to_csv("test_attribute_distribution.csv")

