import pandas as pd

def xoa_trung_lap(s):
  loop = ""
  le = len(s)
  i=1
  while i <= len(s)-1:
    if s[i]==s[i-1] and (i==len(s)-1 or s[i+1]==' '):
      j=i
      loop=s[i]
      while s[j-1] == s[j]:
        loop+=s[j]
        j-=1
      s = s.replace(loop, s[i])
    i+=1
  return s

#load dictionary guid
dic_guid = pd.read_csv('./DicGuid.csv')

def map2good(sen, dic):
  for i in dic.values:
    sen=sen.replace(i[0], i[1])
  return sen

def scanerr(sentence):
  sentence=sentence.lower()
  #chuyen tu tieng viet, teencode, tu kho hieu sang dang de hon
  sentence = map2good(sentence, dic_guid)

  #xoa ky tu nghi la spam
  tem=''
  i=0
  while sentence[i] != '.' and sentence[i] != '!' and i<len(sentence)-1:
    
    if sentence[i] == 'w' or sentence[i] == 'z' or sentence[i]=='f' or sentence[i]=='j':
      tem = sentence[i]
      j=i
      while sentence[j-1] != ' ':
        tem = sentence[j-1]+tem
        j-=1
      while sentence[j+1] != ' ' and j+1 < len(sentence)-1:
        tem = tem + sentence[j+1]
        j+=1
    i+=1

  sentence=sentence.replace(tem, '')
  #xoa ky tu dac biet
  sentence=sentence.replace(":3", '')
  sentence=sentence.replace("<3", '')
  sentence=sentence.replace(":>", '')
  sentence=sentence.replace(":v", '')
  sentence=sentence.replace(":)", '')
  sentence=sentence.replace("=)", '')
  sentence=sentence.replace(")", '')
  sentence=sentence.replace(":(", '')
  sentence=sentence.replace("(", '')
  sentence=sentence.replace("!", '')
  sentence=sentence.replace("?", '')
  sentence=sentence.replace(",", '')
  sentence=sentence.replace("'", '')
  sentence=sentence.replace('"', '')
  sentence=sentence.replace('^', '')
  sentence=sentence.replace('_', '')

  sentence = xoa_trung_lap(sentence)

  return sentence
