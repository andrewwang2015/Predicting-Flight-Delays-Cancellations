import pandas as pd
import numpy as np

df = pd.read_csv('BigData_Pruned_Drop_Delays.csv')
ids = np.where(df['ARRIVAL_DELAY'] > 15)[0]
newColumn = [0 for i in range(len(df))]
for i in ids:
	newColumn[i] = 1
df['BOOL_DELAY'] = newColumn
df.to_csv('all_featuers_boolean.csv')