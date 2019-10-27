import pandas as pd
import numpy as np

dfs = list()
max = list()
for k in range(10):
    df = pd.read_csv('proteins/log/training'+str(k)+'.log')['val_acc']
    dfs.append(df)
    max.append(np.max(df))

print(max)
print(np.mean(max), '+-', np.std(max))

df = pd.read_csv('proteins/log/training' + str(0) + '.log')['val_loss']

for line in df:
    print(line)



