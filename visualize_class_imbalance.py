import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure

df_og = pd.read_csv("AFFACT_balanced/er_AFFACT1_validation_og.csv", index_col=0)
df_reweighed = pd.read_csv("AFFACT_balanced/er_AFFACT1_validation_unbalanced_cube_mean.csv", index_col=0)

output_file("AFFACT_balanced/errors_grouped.html")

attributes = df_og.columns.tolist()

factors = []

for a in attributes:
    factors.append((a, "original"))
    factors.append((a, "reweighed"))

error_types = ["false_positive", "false_negative"]

false_positives1 = df_og.iloc[0].values
false_positives2 = df_reweighed.iloc[0].values
false_positives = []

for i in range(len(false_positives1)):
    false_positives.append(false_positives1[i])
    false_positives.append(false_positives2[i])

false_negatives1 = df_og.iloc[1].values
false_negatives2 = df_reweighed.iloc[1].values
false_negatives = []

for i in range(len(false_positives1)):
    false_negatives.append(false_negatives1[i])
    false_negatives.append(false_negatives2[i])

print(false_positives)
print(false_negatives)

source = ColumnDataSource(data=dict(
    x=factors,
    false_positive=false_positives,
    false_negative=false_negatives
))

p = figure(x_range=FactorRange(*factors), plot_height=1000, plot_width=2000, toolbar_location=None, tools="")

p.vbar_stack(error_types, x='x', width=0.9, alpha=0.5, color=["blue", "red"], source=source,
             legend_label=error_types)

p.y_range.start = 0
p.y_range.end = 7000
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.legend.location = "top_center"
p.legend.orientation = "horizontal"

show(p)

print(factors)

