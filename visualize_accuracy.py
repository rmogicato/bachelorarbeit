import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

df1 = pd.read_csv("error_rates/extracted_attributes_validation_AFFACT2", header=0, index_col=0)
df2 = pd.read_csv("error_rates/extracted_attributes_validation_corrected_cubic_AFFACT2", header=0, index_col=0)
df3 = pd.read_csv("error_rates/extracted_attributes_validation_corrected_square_AFFACT2", header=0, index_col=0)

# set width of bars
barWidth = 0.25

print(df1)

# set heights of bars
bars1 = [np.mean(df1.T.fp_rate.values), np.mean(df1.T.fn_rate.values), np.mean(df1.T.error_rate.values)]
bars2 = [np.mean(df2.T.fp_rate.values), np.mean(df2.T.fn_rate.values), np.mean(df2.T.error_rate.values)]
bars3 = [np.mean(df3.T.fp_rate.values), np.mean(df3.T.fn_rate.values), np.mean(df3.T.error_rate.values)]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#870d04', width=barWidth, edgecolor='white', label='original')
plt.bar(r2, bars2, color='#d1751f', width=barWidth, edgecolor='white', label='std^3*sgn(mean)')
plt.bar(r3, bars3, color='#c4ae06', width=barWidth, edgecolor='white', label='std^2*mean')

# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['false positive', 'false negative', 'total errors'])

# Create legend & Show graphic
plt.legend()
plt.show()

