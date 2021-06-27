import pandas as pd
import numpy as np
from bokeh.io import show
from bokeh.models import ColumnDataSource, FactorRange, Legend
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.io import export_png

error_rate_0 = pd.read_csv("error_rates/final/AFFACT1_testing_no_reweight.csv", header=0, index_col=0)
error_rate_1 = pd.read_csv("error_rates/final/AFFACT1_testing_balanced_square_mean.csv", header=0, index_col=0)
error_rate_2 = pd.read_csv("error_rates/final/AFFACT1_testing_balanced_square_sign.csv", header=0, index_col=0)
error_rate_3 = pd.read_csv("error_rates/final/AFFACT1_testing_balanced_cube_mean.csv", header=0, index_col=0)
error_rate_4 = pd.read_csv("error_rates/final/AFFACT1_testing_balanced_cube_sign.csv", header=0, index_col=0)

attributes = error_rate_1.columns.to_list()
reweigh_methods = ["no reweigh", "reweigh square mean", "reweigh square sign", "reweigh cube mean", "reweigh cube sign"]

method0 = []
method1 = []
method2 = []
method3 = []
method4 = []


def get_unbalanced_error_rate(row):
    accuracy = (row["correct_positives"]+row["correct_negatives"])/(row["correct_positives"]+row["correct_negatives"]+row["false_positives"]+row["false_negatives"])
    return 1-accuracy


def get_balanced_error_rate(row):
    sensitivity = row["correct_positives"] / (row["correct_positives"]+row["false_negatives"])
    specificity = row["correct_negatives"] / (row["false_positives"]+row["correct_negatives"])
    accuracy = (sensitivity + specificity)/2
    return 1-accuracy


for a in attributes:
    method0.append(get_balanced_error_rate(error_rate_0[a]))
    method1.append(get_balanced_error_rate(error_rate_1[a]))
    method2.append(get_balanced_error_rate(error_rate_2[a]))
    method3.append(get_balanced_error_rate(error_rate_3[a]))
    method4.append(get_balanced_error_rate(error_rate_4[a]))

attributes.append("total")
method0.append(np.array(method0).mean())
method1.append(np.array(method1).mean())
method2.append(np.array(method2).mean())
method3.append(np.array(method3).mean())
method4.append(np.array(method4).mean())

data = {
    "attributes": attributes,
    reweigh_methods[0]: method0,
    reweigh_methods[1]: method1,
    reweigh_methods[2]: method2,
    reweigh_methods[3]: method3,
    reweigh_methods[4]: method4,
}

source = ColumnDataSource(data=data)

p = figure(x_range=attributes, y_range=(0, 0.35), plot_height=800, title="Balanced error rates per attribute ",
           toolbar_location=None, tools="")

p.vbar(x=dodge('attributes', -0.25, range=p.x_range), top=reweigh_methods[0], width=0.08, source=source,
       color="#023047", legend_label=reweigh_methods[0])

p.vbar(x=dodge('attributes',  -0.125,  range=p.x_range), top=reweigh_methods[1], width=0.08, source=source,
       color="#219ebc", legend_label=reweigh_methods[1])

p.vbar(x=dodge('attributes',  0,  range=p.x_range), top=reweigh_methods[2], width=0.08, source=source,
       color="#8ecae6", legend_label=reweigh_methods[2])

p.vbar(x=dodge('attributes',  0.125, range=p.x_range), top=reweigh_methods[3], width=0.08, source=source,
       color="#ffb703", legend_label=reweigh_methods[3])

p.vbar(x=dodge('attributes',  0.25, range=p.x_range), top=reweigh_methods[4], width=0.08, source=source,
       color="#fb8500", legend_label=reweigh_methods[4])


p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.legend.location = "top_right"
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = "vertical"

show(p)
export_png(p, filename="balanced_comparison.png")
