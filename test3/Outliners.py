import agate
import agatestats
import pandas as pd
import numpy as np

table = agate.Table.from_csv('XZAB.csv')
# agatestats.patch()

outliers = table.stdev_outliers('Fault', deviations=3, reject=True)
print(outliers)
print(len(outliers.rows))
outliers1 = table.mad_outliers('Fault', deviations=3, reject=False)
print(outliers1)
print(len(outliers1.rows))
# outliers.to_csv('outline.csv')

