#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec


# In[22]:


#import package
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora,models
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from stop_words import get_stop_words
import pandas as pd
import json
import warnings
import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import time
from tqdm import tqdm
import nltk
import matplotlib.pyplot as plt


# In[3]:


import pandas as pd


# In[4]:


documents_consumerreports_1=pd.read_csv(r"C:\Users\eric\Desktop\consumerreports_text.csv",header=0)
documents_consumerreports_1=documents_consumerreports_1.rename(columns={'0':'text'})
documents_consumerreports_1 = documents_consumerreports_1.drop('1', 1)
documents_consumerreports_1=documents_consumerreports_1[:180]


# In[6]:


#documents_consumerreports_1.head()
len(documents_consumerreports_1)


# In[8]:


documents_autofutures=pd.read_csv(r"C:\Users\eric\Desktop\autofutures.csv",header=0)
documents_autofutures.head()
documents_autofutures=documents_autofutures.rename(columns={'0':'text'})
documents_autofutures=documents_autofutures[:500]


# In[9]:


len(documents_autofutures)


# In[10]:


documents_2025AD=pd.read_csv(r"C:\Users\eric\Desktop\2028_new_text.csv",header=0)
documents_2025AD=documents_2025AD.rename(columns={'0':'text'})


# In[11]:


documents_caranddriver=pd.read_csv(r"C:\Users\eric\Desktop\caranddriver_text.csv",header=0)
documents_caranddriver=documents_caranddriver.rename(columns={'0':'text'})


# In[12]:


documents_raw=pd.concat([documents_consumerreports_1,documents_caranddriver,documents_autofutures,documents_2025AD])


# In[13]:


documents_raw=documents_raw.reset_index(drop= True)


# In[14]:


len(documents_raw)


# In[57]:


#불용어사전만들기
en_stop = get_stop_words('en')
en_stop.extend(["self", "automaker","automotive","driving", "car", "autonomous", "vehicle","driverless","driver","drive","drivers","auto","vehicles","cars","automated",'automatic'])


# In[18]:


#모든 대문자를 소문자로
documents_raw = documents_raw.fillna('').astype(str).apply(lambda x: x.str.lower()) 


# In[19]:


#불필요한거제거
documents_raw['text']= documents_raw['text'].map(lambda x: re.sub('\s+', ' ', x))
documents_raw['text']= documents_raw['text'].map(lambda x: re.sub("\'", "", x))


# In[24]:


#lemmatize
Lem = WordNetLemmatizer()
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(Lem.lemmatize(token,pos="n"))
    return result


# In[25]:


processed_docs = documents_raw['text'].map(preprocess)


# In[26]:


#불용어 제거
import numpy as np
def remove_stopwords(, en_stop):
    
    '''
    A function for removing unused word lists from text.
    
    ---parameter---
    series: Pandas Series
    using_word_list: List for using word
    
    '''
    result= []
    for doc in series:
        text= [word for word in doc if word not in en_stop]
        result.append(text)
    return result


# In[27]:


stopped_processed_docs =remove_stopwords(processed_docs,en_stop)


# In[30]:


# #제거되었는지확인
# stopped_processed_docs[0]


# In[31]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(stopped_processed_docs, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[stopped_processed_docs], threshold=100)
 
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
 
# See trigram example0
print(trigram_mod[bigram_mod[stopped_processed_docs[0]]])


# In[32]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


# In[33]:


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[34]:


# Form Bigrams
data_words_bigrams = make_bigrams(stopped_processed_docs)


# In[35]:


type(data_words_bigrams)


# In[242]:


#불용어사전만들기
en_stop = get_stop_words('en')
en_stop.extend(["self", "automaker","automotive","driving", "car", "autonomous", "vehicle","driverless","driver","drive","auto"])


# In[37]:


from nltk.tokenize import word_tokenize, sent_tokenize


# In[230]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[231]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None
def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:            
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


# In[246]:


#dataframe to sent_list
sent_text_list=[]
result=[]
for i in range(len(documents_raw)):
    sent_text=sent_tokenize(documents_raw.text[i])
    singles = [lemmatize_sentence(sent) for sent in sent_text]
    result=[word_tokenize(sentence) for sentence in singles]
    sent_text_list.append(result)


# In[248]:


sent_text_list[0][0]


# In[290]:


result_a= []
input_data=[]
from tqdm import tqdm
for i in tqdm(range(len(sent_text_list))):
    for sent in sent_text_list[i]:
        text= [word for word in sent if word not in en_stop]
        text=[x.lower() for x in text]
        text=[re.sub("[^a-zA-Z]", " ", j) for j in text]
        result_a.append(text)
    input_data.append(result_a)


# In[357]:


input_data[0]


# In[310]:


type(input_data[0][0])


# In[356]:


input_data[0]


# In[314]:


from gensim.models import word2vec
model = Word2Vec(input_data[0], size=100, window=5, min_count=3, workers=4)


# In[315]:


# model.wv.vocab.keys()


# In[355]:


model.most_similar('fully')


# In[ ]:




