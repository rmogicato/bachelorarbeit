import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.io import export_png

"""
This files can compare 3 detailed error rates. The detail error rate can be obtained by calling calculate_accuracy(),
which provides us with 2 output files, the second of which is the detailed error rate.
We generate 40 groups of bars, each group containing 3 stacked bars,
each bar contains the false positive error rate and the false negative error rate.
"""

df_og = pd.read_csv("final_error_rates/detailed/AFFACT-A_og.csv", index_col=0, header=0)
df_reweighed_af = pd.read_csv("final_error_rates/detailed/AFFACT-A_AF_square_sign_balanced.csv", index_col=0, header=0)
df_reweighed_gt = pd.read_csv("final_error_rates/detailed/AFFACT-A_GT_square_sign_balanced.csv", index_col=0, header=0)

output_file("AFFACT_balanced/errors_grouped.html")

attributes = df_og.index.tolist()
attributes_clean = []
for a in attributes:
    attributes_clean.append(a.replace("_", " "))
factors = []

for a in attributes_clean:
    factors.append((a, "original"))
    factors.append((a, "clustered"))
    factors.append((a, "ground truth"))

error_types = ["false_positive", "false_negative"]

false_positives_og = df_og["false_positive_rate"].values
false_positives_af = df_reweighed_af["false_positive_rate"].values
false_positives_gt = df_reweighed_gt["false_positive_rate"].values
false_positives = []

for i in range(len(false_positives_og)):
    false_positives.append(false_positives_og[i])
    false_positives.append(false_positives_af[i])
    false_positives.append(false_positives_gt[i])


false_negatives1 = df_og["false_negative_rate"].values
false_negatives2 = df_reweighed_af["false_negative_rate"].values
false_negatives3 = df_reweighed_gt["false_negative_rate"].values
false_negatives = []

for i in range(len(false_positives_og)):
    false_negatives.append(false_negatives1[i])
    false_negatives.append(false_negatives2[i])
    false_negatives.append(false_negatives3[i])


source = ColumnDataSource(data=dict(
    x=factors,
    false_positive=false_positives,
    false_negative=false_negatives
))

p = figure(x_range=FactorRange(*factors), plot_height=1160, plot_width=1660, toolbar_location=None, tools="")


p.vbar_stack(error_types, x='x', width=0.9, alpha=0.5, color=["blue", "red"], source=source)

p.y_range.start = 0
p.y_range.end = 0.7
p.x_range.range_padding = 0
p.yaxis.major_label_orientation = np.pi/2
p.xaxis.major_label_orientation = np.pi/2
p.xaxis.group_label_orientation = np.pi/2
p.xgrid.grid_line_color = None
show(p)
export_png(p, filename="plot.png")