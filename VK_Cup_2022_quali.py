#
#installing Packages
#
#!pip install transformers
#!pip install spacy
#!python -m spacy download ru_core_news_sm

#
import requests

def get(url, to):
    r = requests.get(url)
    with open(to, 'wb') as f:
        f.write(r.content)
    print(r.status_code)

get('https://raw.githubusercontent.com/Lojaleto/loader/main/loader.py', './loader.py')
get('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/bert.py', './bert.py')

#
import pandas as pd
import numpy as np
import torch
import spacy
import re
import matplotlib.pyplot as plt
import seaborn as sns

from spacy.lang.ru.examples import sentences 
from loader import download
from bert import BertClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

#
#get data
#
download('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/subsets/VK_Cup_2022_quali/lemma.csv',
         './subsets/VK_Cup_2022_quali/lemma.csv')
download('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/subsets/VK_Cup_2022_quali/lemma_test.csv',
         './subsets/VK_Cup_2022_quali/lemma_test.csv')
download('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/datasets/VK_Cup_2022_quali/train.csv',
         './datasets/VK_Cup_2022_quali/train.csv')
download('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/datasets/VK_Cup_2022_quali/test.csv',
         './datasets/VK_Cup_2022_quali/test.csv')
download('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/datasets/VK_Cup_2022_quali/sample_submission.csv',
         './datasets/VK_Cup_2022_quali/sample_submission.csv')
download('https://github.com/Lojaleto/VK_Cup_2022_quali/raw/main/subsets/VK_Cup_2022_quali/bert.pt',
         './subsets/VK_Cup_2022_quali/bert.pt')

#
df = {}

#df["train"] = pd.read_csv('./datasets/VK_Cup_2022_quali/train.csv')
#df["test"] = pd.read_csv('./datasets/VK_Cup_2022_quali/test.csv')
df["sample_submission"]  = pd.read_csv('./datasets/VK_Cup_2022_quali/sample_submission.csv')

#
#lemmatization
#
'''
nlp = spacy.load('ru_core_web_sm', disable=['parser', 'ner'])
def spacy_fn(sentence):
    doc = nlp(re.sub(r'[^0-9a-zA-Zа-яА-ЯёЁ]', ' ', sentence).lower())
    return " ".join([token.lemma_ for token in doc])
    
df["train"]['text'] = df["train"]['text'].apply(lambda x: spacy_fn(x))
df["train"].to_csv('./subsets/VK_Cup_2022_quali/lemma.csv', index=False)

df["test"]['text'] = df["test"]['text'].apply(lambda x: spacy_fn(x))
df["test"].to_csv('./subsets/VK_Cup_2022_quali/lemma_test.csv', index=False)
'''
pass

#загрузим уже лемматизированные данные
df["train"] = pd.read_csv('./subsets/VK_Cup_2022_quali/lemma.csv')
df["test"] = pd.read_csv('./subsets/VK_Cup_2022_quali/lemma_test.csv')

#
#initialize BERT classifier
#
classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=13,
        epochs=33,
        model_save_path='./subsets/VK_Cup_2022_quali/bert.pt'
)

keys_target = {'athletics':   0,
               'autosport':   1,
               'basketball':  2,
               'boardgames':  3,
               'esport':      4,
               'extreme':     5,
               'football':    6,
               'hockey':      7,
               'martial_arts':8,
               'motosport':   9,
               'tennis':      10,
               'volleyball':  11,
               'winter_sport':12}
               
features = df["train"].drop('category', axis=1)
target = df["train"].drop('text', axis=1)

target['category'] = target['category'].apply(lambda x: keys_target[x])

X = {}
y = {}

X["train"], X["valid"], y["train"], y["valid"] = train_test_split(features, target, test_size=0.3,
                                                                stratify=target, random_state=12345)
X["valid"], X["test"], y["valid"], y["test"] = train_test_split(features, target, test_size=0.2,
                                                                stratify=target, random_state=12345)

display(pd.DataFrame([[X["train"].shape, X["valid"].shape, X["test"].shape],
                      [y["train"].shape, y["valid"].shape, y["test"].shape]],
                     columns=['train', 'valid', 'test'],
                     index=['X', 'y']))

#
#prepare data and helpers for train and evlauation
#
classifier.preparation(
        X_train=list(X["train"]['text']),
        y_train=list(y["train"]['category']),
        X_valid=list(X["valid"]['text']),
        y_valid=list(y["valid"]['category'])
    )

#
#train loop
#
#classifier.train()

#
#check test data
#
#загрузим уже дообученную модель
classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        model_save_path='./subsets/VK_Cup_2022_quali/bert.pt',
        model_load=True
)

#
#prediction
#
df["train_test"] = X["test"]
df["train_test"]['category'] = y["test"]['category']
df["train_test"] = df["train_test"][['oid', 'category', 'text']]

#инвертируем словарь
keys_target = {value: key for key, value in keys_target.items()}

df["train_test"][['pred', 'prob']] = [classifier.predict(t, proba=True) for t in list(df["train_test"]['text'])]

df["train_test"]['pred'] = df["train_test"]['pred'].apply(lambda x: keys_target[x])
df["train_test"]['category'] = df["train_test"]['category'].apply(lambda x: keys_target[x])

df["train_test"].head()

###
plt.rcParams['figure.figsize'] = [16, 7]
sns.histplot(df["train_test"]['prob'],
             color="gray", label="prob", bins=100)
plt.legend() 
plt.show()


#df["filt"] = df["train_test"]
#установим приемлемую вероятность
df["filt"] = df["train_test"][df["train_test"]['prob'] > 0.9]

precision, recall, f1score = precision_recall_fscore_support(df["filt"]['category'],
                                                             df["filt"]['pred'], average='macro')[:3]
print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')

#
#result
#
df["res"] = pd.DataFrame(df["test"]['oid'].copy())

df["res"][['category', 'prob']] = [classifier.predict(t, proba=True) for t in list(df["test"]['text'])]

df["res"]['prob'][df["res"]['prob'] < 0.9].count() / df["res"]['category'].count()

df["res"]['category'] = df["res"]['category'].apply(lambda x: keys_target[x])
df["res"].loc[df["res"]['prob'] < 0.9, 'category'] = ''

df["res"].head()

df["res"] = df["res"].drop('prob', axis=1)

df["res"].to_csv('./subsets/VK_Cup_2022_quali/res.csv', index=False)

df["res"].head()

