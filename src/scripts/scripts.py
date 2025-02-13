import pandas as pd

df = pd.read_csv('../data/MinimaxVSMinimaxPruned1Sec.csv')

print(df.select_dtypes(include='number').mean())