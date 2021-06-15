import pandas as pd
import numpy as np
from bokeh.io import show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import dodge

error_rate_1 = pd.read_csv("error_rates/er_new_testing_AFFACT1.txt", header=0, index_col=0)
error_rate_2 = pd.read_csv("error_rates/er_arcface_testing_reweighed_square_mean_AFFACT1.txt", header=0, index_col=0)
error_rate_3 = pd.read_csv("error_rates/er_arcface_testing_reweighed_cube_mean_AFFACT1.txt", header=0, index_col=0)

attributes = error_rate_1.columns.to_list()
reweigh_methods = ["no_reweigh", "reweigh with arcface", "reweigh ground truth"]
method1 = []
method2 = []
method3 = []


def get_unbalanced_error_rate(row):
    accuracy = (row["correct_positives"]+row["correct_negatives"])/(row["correct_positives"]+row["correct_negatives"]+row["false_positives"]+row["false_negatives"])
    return 1-accuracy


def get_balanced_error_rate(row):
    sensitivity = row["correct_positives"] / (row["correct_positives"]+row["false_negatives"])
    specificity = row["correct_negatives"] / (row["false_positives"]+row["correct_negatives"])
    accuracy = (sensitivity + specificity)/2
    return 1-accuracy


for a in attributes:
    method1.append(get_balanced_error_rate(error_rate_1[a]))
    method2.append(get_balanced_error_rate(error_rate_2[a]))
    method3.append(get_balanced_error_rate(error_rate_3[a]))

attributes.append("total")
method1.append(np.array(method1).mean())
method2.append(np.array(method2).mean())
method3.append(np.array(method3).mean())

data = {
    "attributes": attributes,
    reweigh_methods[0]: method1,
    reweigh_methods[1]: method2,
    reweigh_methods[2]: method3
}

source = ColumnDataSource(data=data)

p = figure(x_range=attributes, y_range=(0, 0.5), plot_height=1000, title="Balanced error rates per attribute ",
           toolbar_location=None, tools="")

p.vbar(x=dodge('attributes', -0.25, range=p.x_range), top=reweigh_methods[0], width=0.2, source=source,
       color="#c9d9d3", legend_label=reweigh_methods[0])

p.vbar(x=dodge('attributes',  0.0,  range=p.x_range), top=reweigh_methods[1], width=0.2, source=source,
       color="#718dbf", legend_label=reweigh_methods[1])

p.vbar(x=dodge('attributes',  0.25, range=p.x_range), top=reweigh_methods[2], width=0.2, source=source,
       color="#e84d60", legend_label=reweigh_methods[2])


p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.legend.location = "top_right"
p.legend.orientation = "horizontal"
p.xaxis.major_label_orientation = "vertical"

show(p)
