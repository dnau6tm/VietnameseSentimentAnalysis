from transformers import RobertaForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
from processdata import scanerr
import torch
import py_vncorenlp

py_vncorenlp.download_model(save_dir='./')

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./')

tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base")

model = RobertaForSequenceClassification.from_pretrained("./out_model")
model.cuda()

test_df = pd.read_csv("./data_phase_2.txt", sep="\t", header=None)

ans = []
for sentence in test_df[0]:
     be = sentence
     sentence = scanerr(sentence)
     sen_af = rdrsegmenter.word_segment(sentence)
     input_ids = torch.tensor([tokenizer.encode(sen_af[0])])


     with torch.no_grad():
          out = model(input_ids)
          if np.argmax(out.logits.softmax(dim=-1).tolist())==0:
               ans.append('negative')
          elif np.argmax(out.logits.softmax(dim=-1).tolist())==1:
               ans.append('positive')
          else:
               ans.append('neutral')


with open("/content/submit.txt", "w") as txt_file:
     for line in ans:
          txt_file.write("".join(line) + "\n")

      