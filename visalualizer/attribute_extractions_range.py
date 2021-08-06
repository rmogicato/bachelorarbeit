import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot

"""
This file generates a boxplot to give us an idea how large ranges of extracted values are.

"""


plt.rcParams["font.family"] = "Palatino Linotype"

df_validation_1 = pd.read_csv("../extractions/new_validation_AFFACT1.txt", header=0, index_col=0)
print(df_validation_1)
attributes = df_validation_1.columns.to_list()[:40]
attributes_clean = []
for a in attributes:
    attributes_clean.append(a.replace("_", " "))


mins = []
maxs = []
data = []
for a in attributes:
    arr = df_validation_1[a].values
    mins.append(np.min(arr))
    maxs.append(np.max(arr))
    data.append(arr)
print(mins)

fig = plt.figure(figsize =(10, 6))

ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True,
                notch='True', vert=1)
ax.set_xticklabels(attributes_clean)
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.3)
plt.title("Distribution of Extracted Values from AFFACT-B")
plt.show()
fig.savefig("boxplot.pdf")
