import pandas as pd
import py_vncorenlp
from processdata import scanerr

py_vncorenlp.download_model(save_dir='./')

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./')

df = pd.read_csv("./train.txt", sep="\t", header=None)

def process_data2(row):
  sentence = scanerr(row[1])
  sen_af = rdrsegmenter.word_segment(sentence)
  label = 0
  if row[0] == 'negative':
    label = 0
  elif row[0] == 'positive':
    label = 1
  else:
    label = 2
  return sen_af[0], label

train_df=df[:4000].dropna(axis=0)
test_df = df[4000:].dropna(axis=0)

arr1 = []
arr2 = []
for i in train_df.values:
  x, y = process_data2(i)
  arr1.append(x)
  arr2.append(y)

d={'text':arr1, 'label': arr2}
df_csv = pd.DataFrame(data=d)

arr3 = []
arr4 = []
for i in test_df.values:
  x, y = process_data2(i)
  arr3.append(x)
  arr4.append(y)

dt={'text':arr3, 'label': arr4}
dft_csv = pd.DataFrame(data=dt)

df_csv.to_csv("./train.csv", index=False)
dft_csv.to_csv("./test.csv", index=False)


