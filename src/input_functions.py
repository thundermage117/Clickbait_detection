import numpy as np
import pandas as pd
import torch
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def extractData(df2, labels):
  label_values = np.where(labels.iloc[:, 4]=='clickbait', 1, 0)
  heading = df2.iloc[: ,4]
  body = df2.iloc[:, 5]
  return heading, body, label_values

def getMaxSentLength(a):
  max_sentence_length = 0
  for i in range(len(a)):
    if (max_sentence_length < len(a[i])):
      max_sentence_length = len(a[i])

  return max_sentence_length

def getVecForm(sentence, word2vecLength, word_vec, max_sentLength):
  sentence_vec = torch.zeros((max_sentLength, word2vecLength))
  # sentence_vec = np.zeros((max_sentLength, word2vecLength))
  for i in range(len(sentence)):
    sentence_vec[i] = torch.FloatTensor(word_vec[sentence[i]].copy())
    # sentence_vec[i] = np.array(word_vec[sentence[i]].copy())

  return sentence_vec

def getVecDataFrame(heading, body,labels_val, NormLength, word2vecLength, word_vec):
  head_vec = torch.zeros((len(heading), NormLength, word2vecLength))
  # head_vec = np.zeros((len(heading), NormLength, word2vecLength))
  # label_vec=torch.zeros((len(heading)))
  for i in range(len(heading)):
    head_vec[i] = getVecForm(heading[i], word2vecLength, word_vec, NormLength)

  body_vec = torch.zeros((len(body), NormLength, word2vecLength))
  # body_vec = np.zeros((len(body), NormLength, word2vecLength))
  for i in range(len(body)):
    body_vec[i] = getVecForm(body[i], word2vecLength, word_vec, NormLength)  

  headBody_vecDF = pd.DataFrame(list(zip(head_vec, body_vec,labels_val)), columns =['Heading', 'Body','Labels'])
  return headBody_vecDF,head_vec,body_vec

regexp = nltk.tokenize.RegexpTokenizer('\w+')
word_lem= nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")

# Accepts string type variable
def stringProc(test):
  test = test.lower()
  test = word_lem.lemmatize(test)
  test_tokens = regexp.tokenize(test)
  test_remStop = [word for word in test_tokens if word not in stopwords]
  return test_remStop

# This function returns the tokenized form for a head and body vector
def tokenizer(head_text, body_text, word2vecLength, word_vec, max_sentLength):

  if (type(head_text) != str):
    head_text = head_text[0]
  if (type(body_text) != str):
    body_text = body_text[0]

  head_text_proc = stringProc(head_text)
  body_text_proc = stringProc(body_text)

  head_vec_form = getVecForm(head_text_proc, word2vecLength, word_vec, max_sentLength)
  body_vec_form = getVecForm(body_text_proc, word2vecLength, word_vec, max_sentLength)

  return head_vec_form, body_vec_form