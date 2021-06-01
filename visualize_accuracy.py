import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 7)

df1 = pd.read_csv("error_rates/er_arcface_validation_arcface_reweighed_cube_sign_AFFACT1.txt", header=0, index_col=0)
df2 = pd.read_csv("error_rates/er_new_validation_AFFACT1.txt", header=0, index_col=0)
df3 = pd.read_csv("error_rates/validation_reweighed_cube_sign.txt", header=0, index_col=0)

dfs = [df1, df2, df3]
# set width of bars
barWidth = 0.25

false_positive_rates = []
false_negative_rates = []
balanced_total = []

for df in dfs:
    # summing up the numbers of all attributes
    column_list = list(df)
    df["sum"] = df[column_list].sum(axis=1)

    fp = df.at["false_positives", "sum"]
    tp = df.at["correct_positives", "sum"]
    fn = df.at["false_negatives", "sum"]
    tn = df.at["correct_negatives", "sum"]

    # false positive rate
    fp_rate = fp / (fp + tp)
    false_positive_rates.append(fp_rate)

    # false negative rate
    fn_rate = fn / (fn + tn)
    false_negative_rates.append(fn_rate)

    # balanced error rate

    balanced_error_rate = 0.5 * (fp_rate + fn_rate)
    balanced_total.append(balanced_error_rate)


# set heights of bars
bars1 = [false_positive_rates[0], false_negative_rates[0], balanced_total[0]]
bars2 = [false_positive_rates[1], false_negative_rates[1], balanced_total[1]]
bars3 = [false_positive_rates[2], false_negative_rates[2], balanced_total[2]]

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

