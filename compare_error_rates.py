import pandas as pd
import numpy as np
from bokeh.io import output_file, show
from bokeh.plotting import figure

df_gt = pd.read_csv("final_error_rates/AFFACT-B_GT_improvement_square_mean.csv", header=0, index_col=0)
df_af = pd.read_csv("final_error_rates/AFFACT-B_improvement_square_mean.csv")

attributes = df_gt.index.to_list()
