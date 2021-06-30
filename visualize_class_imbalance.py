import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import dodge


df_og = pd.read_csv("error_rates/final/gt_AFFACT1_testing_no_reweight.csv", index_col=0)
df_reweighed1 = pd.read_csv("error_rates/final/gt_AFFACT1_testing_balanced_square_sign.csv", index_col=0)
df_reweighed2 = pd.read_csv("error_rates/final/af_AFFACT1_testing_balanced_square_sign.csv", index_col=0)

output_file("AFFACT_balanced/errors_grouped.html")

attributes = df_og.columns.tolist()
attributes.append("Average")
factors = []

for a in attributes:
    factors.append((a, "original"))
    factors.append((a, "arcface"))
    factors.append((a, "ground truth"))

error_types = ["false_positive", "false_negative"]

false_positives1 = df_og.iloc[0].values
false_positives2 = df_reweighed1.iloc[0].values
false_positives3 = df_reweighed2.iloc[0].values
false_positives = []

for i in range(len(false_positives1)):
    false_positives.append(false_positives1[i])
    false_positives.append(false_positives2[i])
    false_positives.append(false_positives3[i])


false_positives.append(np.mean(false_positives1))
false_positives.append(np.mean(false_positives2))
false_positives.append(np.mean(false_positives3))


false_negatives1 = df_og.iloc[1].values
false_negatives2 = df_reweighed1.iloc[1].values
false_negatives3 = df_reweighed2.iloc[1].values
false_negatives = []

for i in range(len(false_positives1)):
    false_negatives.append(false_negatives1[i])
    false_negatives.append(false_negatives2[i])
    false_negatives.append(false_negatives3[i])

false_negatives.append(np.mean(false_negatives1))
false_negatives.append(np.mean(false_negatives2))
false_negatives.append(np.mean(false_negatives3))
print(false_positives)
print(false_negatives)

source = ColumnDataSource(data=dict(
    x=factors,
    false_positive=false_positives,
    false_negative=false_negatives
))

p = figure(x_range=FactorRange(*factors), plot_height=1000, plot_width=2000, toolbar_location=None, tools="",
           title="Total errors of balanced reweighting function square sign")


p.vbar_stack(error_types, x='x', width=0.9, alpha=0.5, color=["blue", "red"], source=source, legend_label=error_types)

p.y_range.start = 0
p.y_range.end = 7000
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = np.pi/2
p.xaxis.group_label_orientation = np.pi/2
p.xgrid.grid_line_color = None
p.legend.location = "top_center"
p.legend.orientation = "horizontal"

show(p)
