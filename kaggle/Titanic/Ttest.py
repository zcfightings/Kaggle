import pandas as pd

data = [[1,2,3],[4,5,6]]
index = ['p',3]
columns=['a','b','c']
df = pd.DataFrame(data=data, index=index, columns=columns)
print(df.loc['p'])
