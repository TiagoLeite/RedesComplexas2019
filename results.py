import pandas as pd
import numpy as np

logs = list()
dataset = input('Dataset: ')

for k in range(10):
    df = pd.read_csv(dataset + '/log/training' + str(k) + '.log')
    logs.append(df)


max_acc = [np.max(log['val_acc']) for log in logs]
max_precision = [np.max(log['val_precision']) for log in logs]
max_recall = [np.max(log['val_recall']) for log in logs]
max_fscore = [np.max(log['val_f_score']) for log in logs]


model_mean_acc = np.mean(max_acc)
model_std_acc = np.std(max_acc)

model_mean_precision = np.mean(max_precision)
model_std_precision = np.std(max_precision)

model_mean_recall = np.mean(max_recall)
model_std_recall = np.std(max_recall)

model_mean_fscore = np.mean(max_fscore)
model_std_fscore = np.std(max_fscore)


print('\nFinal val acc for', dataset)
print(model_mean_acc, '+-', model_std_acc)

print('\nFinal val prec for', dataset)
print(model_mean_precision, '+-', model_std_precision)

print('\nFinal val recall for', dataset)
print(model_mean_recall, '+-', model_std_recall)

print('\nFinal val f_score for', dataset)
print(model_mean_fscore, '+-', model_std_fscore)





