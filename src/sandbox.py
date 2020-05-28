import pandas as pd
import numpy as np

"""
    For testing purposes!
"""
def sep():
    print("===========")

df = pd.DataFrame(np.array(
    [
        [1, 2, None],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, None]]),
                  columns=['a', 'b', 'c'])

print(df)

sep()

filter = df.isin({'a': [1, 4], 'b': [5, 8]})
print(filter)
print(filter.apply(pd.Series.value_counts, axis=1))

sep()

print(df['c'].isnull())

print(len(df[
              (df['c'].isnull() == False) & (df['a'] == 1) & (df['b'] == 2)
          ]))

