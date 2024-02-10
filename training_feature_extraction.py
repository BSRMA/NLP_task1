# Import necessary libraries from the Natural Language Toolkit (nltk) for text processing.
import nltk
import os
import re
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize

# Import pandas for data manipulation and analysis
import pandas as pd
import numpy as np
# Download essential nltk modules for text processing
nltk.download('averaged_perceptron_tagger')  # Part-of-speech tagging
nltk.download('stopwords') # Common English stopwords
nltk.download('punkt') # Tokenizer
nltk.download('wordnet') # Lexical database for English words

# Import the Natural Language Toolkit (nltk) library for text processing
import nltk

# Import the 'words' corpus from nltk, which contains a list of English words
from nltk.corpus import words

# Download the 'words' corpus if not already present in the nltk data directory
nltk.download('words')

# Create an instance of WordNetLemmatizer from nltk for lemmatization of words
lt = WordNetLemmatizer()

# Create a set of English stopwords using nltk's stopwords corpus
stop_words = set(stopwords.words('english'))

# Create a set of English words from the 'words' corpus for reference
english_words = set(words.words())

#Spacy for sentence tokenization
#!pip install spacy
import spacy
nlp = spacy.load("en_core_web_sm")



#Extracting sentences from textfile

def get_sentences(text):
  """ using spacy to split textfile into sentences"""
  try:
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
      sentences.append(sent.text)
    return sentences
  except:
    return sent_tokenize(text)


#Removing sentences of length 3 or less
def remove_short_sentences(Sentences):
  """input: a list of sentences
     output: a list of sentences, each of length more than 3 words"""
  temp_sent = []
  for sent in Sentences:
    words = word_tokenize(sent)
    if len(words) > 3:
      temp_sent.append(sent)
  return temp_sent

#remove web addresses
def remove_addresses_sent(text):
  clean_text = re.sub(r"\S*www.\S+", "", text)
  clean_text = re.sub(r"\S*WWW.\S+", "", clean_text)
  clean_text = re.sub(r"\S*.com\S+", "", clean_text)
  clean_text = re.sub(r"\S*.co\S+", "", clean_text)
  clean_text = re.sub(r"\S*.COM\S+", "", clean_text)
  clean_text = re.sub(r"\S*.gov\S+", "", clean_text)
  clean_text = re.sub(r"\S*.biz\S+", "", clean_text)
  clean_text = re.sub(r"\S*.org\S+", "", clean_text)
  return clean_text

def remove_addresses(sentences):
  for i in range(len(sentences)):
    sentences[i] = remove_addresses_sent(sentences[i])
  return sentences  
  


def remove_special_characters(sentences):
  temp_text = []
  for sent in sentences:
    temp = ''
    for char in sent:
      if char.isalnum() or char == ' ' or char == '\n':
        temp += char
    temp_text.append(temp)
  return temp_text

def remove_numerical_values(sentences):
  temp_sent = []
  for sent in sentences:
    words = word_tokenize(sent)
    temp = ''
    for word in words:
      if word.isalpha():
        temp += word+' '
    temp_sent.append(temp)
  return temp_sent



#POS tagging, stop word removal, lemmatization
def preprocessing(sentences):
  """input: A list of sentences
     output: the sentences,  a list of lists with each list containing tuples of each words and POS tags"""
  cleaned_sentences = []
  all_sentences = []
  for i in range(len(sentences)):
    sent = sentences[i]
    words = word_tokenize(sent)
    temp = []
    for word in words:
      if word.isalpha():
        temp.append(word)
    if len(temp):
      tagged_words = nltk.pos_tag(temp)
      temp = []
      for tup in tagged_words:
        if tup[0] not in stop_words:
          tup = (lt.lemmatize(tup[0].lower()), tup[1])
          temp.append(tup)
    all_sentences.append(sent)
    cleaned_sentences.append(temp)

  return all_sentences, cleaned_sentences, len(all_sentences)

def tup_sent(cleaned_sentences):
  """input: a list of lists with each list containing tuples of each words and POS tags
     output: a list of strings with each string containing the words of corresponding list of tuples"""
  sentences = []
  for sent in cleaned_sentences:
    temp = ''
    for tup in sent:
      if len(temp) == 0:
        temp += tup[0]
      else:
        temp += ' '+tup[0]
    sentences.append(temp)
  return sentences


#sent features: 1. tf-isf
def word_freq(sentences):
  """
  Input: A list of sentences (sentences).
  Output: A dictionary (word_dict) where keys are unique words in the sentences,
  and values are the frequency of each word in the entire set of sentences.
  """
  word_dict = {}
  for sent in sentences:
    words = list(set(word_tokenize(sent)))
    for word in words:
      if word not in word_dict:
        word_dict[word] = 1
      else:
        word_dict[word] += 1
  return word_dict

def tf_isf(sentences):
  """
  Input: A list of sentences (sentences).
  Output: A list containing the TF-ISF scores for each sentence.
  """
  word_dict = word_freq(sentences)
  n_sent = len(sentences)
  keys = word_dict.keys()
  tf_isf_score_pre_sentence = []

  for key in keys:
    x = word_dict[key]
    x = math.log(float(n_sent)/x)
    word_dict[key] = x

  for i in range(n_sent):
    sent_val = 0
    sent = sentences[i]
    temp = {}
    words = word_tokenize(sent)
    if len(words) == 0:
      tf_isf_score_pre_sentence.append(1)
      continue
    for word in words:
      if word not in temp:
        temp[word] = 1
      else:
        temp[word] += 1
    keys = temp.keys()
    for key in keys:
      tf = float(temp[key])/len(words)
      isf = word_dict[key]
      sent_val += (tf*isf)
    sent_val /= len(words)
    tf_isf_score_pre_sentence.append(sent_val)
  return tf_isf_score_pre_sentence


#sent features: 2. capital words
def capital_or_not(word):
  for i in range(len(word)):
    if word[i].isupper() == True and i != 0:
      return 1
  return 0

def Capital_letters(real_sentences):
  """
  Input: A list of sentences (sentences).
  Output: Normalized capital word score for each sentence
  """
  cp_score = []
  for sent in real_sentences:
    if len(sent) == 0:
      cp_score.append(0)
      continue
    temp = 0
    words = word_tokenize(sent)
    for word in words:
      if word.isalpha() and capital_or_not(word) == 1:
        if word.isupper():
          temp += 2
        else:
          temp += 1
    temp = float(temp)/len(words)
    cp_score.append(temp)
  max_val = max(cp_score)
  for i in range(len(cp_score)):
    cp_score[i] = float(cp_score[i])/max_val
  return cp_score


#sent features: 3. similarity with other sentences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity_sent(sentences):
  """
  Input: A list of sentences (sentences).
  Output: Normalized cosine similarity score for each sentence
  """
  sim_score_sent = []
  vectorizer = CountVectorizer().fit_transform(sentences) # CountVectorizer to convert sentences to a bag-of-words representation
  sim_matrix = cosine_similarity(vectorizer, vectorizer)
  for i in range(len(sentences)):
    x = sum(sim_matrix[i])/float(len(sim_matrix[0]))
    sim_score_sent.append(x)
  max_val = max(sim_score_sent)
  for i in range(len(sim_score_sent)):
    x = sim_score_sent[i]/max_val
    sim_score_sent[i] = 1 - x
  return sim_score_sent


#sent features: 4. keywords
keywords = ['financial statement',
             'annual report',
             'growth',
             'audit committee',
             'executive director',
             'chairman',
             'stock',
             'shareholder',
             'stockholder',
             'dividend',
             'share',
             'acquisition',
             'year end',
             'this year',
             'annual report account',
             'board of directors',
             'million',
             'account',
             'production',
             'revenue',
             'company',
             'business',
             'plc',
             'profit']
keywords2 = ['cost',
             'result',
             'strategy',
             'service',
             'management']


def keyword_sent(sentences):
  """
  Input: A list of sentences (sentences).
  Output: Normalized key word score for each sentence
  """
  k_score_sent = []
  for sent in sentences:
    if len(sent) == 0:
      k_score_sent.append(0)
      continue
    score = 0
    words = word_tokenize(sent)
    for kw in keywords:
      if kw in sent:
        score += 1
    score = float(score)/len(words)
    k_score_sent.append(score)
  max_val = max(k_score_sent)
  for i in range(len(k_score_sent)):
    k_score_sent[i] = float(k_score_sent[i])/max_val
  return k_score_sent


#sent features: 5. content words
def Content_word(cleaned_sentences):
  """
  Input: A list of sentences (sentences).
  Output: Normalized token word score for each sentence
  """
  noun_verb = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
  adjective_adverb = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS"]
  content_score_sent = []
  for sent in cleaned_sentences:
    if len(sent) == 0:
      content_score_sent.append(0)
      continue
    score = 0
    for tup in sent:
      if tup[1] in noun_verb:
        score += 2
      elif tup[1] in adjective_adverb:
        score += 1
    score = float(score) / len(sent)
    content_score_sent.append(score)
  max_val = max(content_score_sent)
  for i in range(len(content_score_sent)):
    content_score_sent[i] = float(content_score_sent[i])/max_val
  return content_score_sent


def get_sent_features(text):
  """
  Input: Text file
  Output: The sentences and 5 features for each sentence
  """
  sentences = get_sentences(text)
  sentences = remove_short_sentences(sentences)
  sentences_0 = remove_addresses(sentences)
  sentences_1 =  remove_special_characters(sentences_0)
  sentences_2 =  remove_numerical_values(sentences_1)

  sent, tuples_sent, N = preprocessing(sentences_2)
  cleared_sent = tup_sent(tuples_sent)

  tfisf = tf_isf(cleared_sent)
  cap = Capital_letters(sent)
  sim = similarity_sent(cleared_sent)
  kw = keyword_sent(cleared_sent)
  cont = Content_word(tuples_sent)

  features = []
  for i in range(N):
    temp = [tfisf[i], cap[i], sim[i], kw[i], cont[i]]
    features.append(temp)
  return sentences, features, [tfisf, cap, sim, kw, cont]


def get_summary_file_name(filename, contents2):
  x = filename.split('.')
  temp = []
  for file in contents2:
    y = file.split('_')
    if x[0] == y[0]:
      temp.append(file)
  return temp

def save_csv(text, name, sum_file_texts, folder_path):
  a, b, c = get_sent_features(text)
  
  keys = sum_file_texts.keys()
  sum_sents = []
  for key in keys:
    sentences = sent_tokenize(sum_file_texts[key])
    sum_sents.extend(sentences)
  sum_sents = list(set(sum_sents))
  
  clas = []
  for x in a:
    if x in sum_sents:
      clas.append(1)
    else:
      clas.append(0)
  
  data_ar = {'Sentences': a,
        'tfisf': c[0],
        'capital': c[1],
        'sent_similarity': c[2],
        'keyword': c[3],
        'content_words': c[4],
        'class': clas}
  df = pd.DataFrame(data_ar)
  training_feat_file_name = name+'.csv'
  training_feat_file_path = os.path.join(folder_path, training_feat_file_name)
  df.to_csv(training_feat_file_path, escapechar='\\')
  
  
annual_report_path = 'training/annual_reports'
summary_path = 'training/gold_summaries'
contents1 = os.listdir(annual_report_path)
contents2 = os.listdir(summary_path)
j = 0

folder_path = "DATA/training_features"
try:
  os.mkdir(folder_path)
except:
  print("File is already there.")


for x in contents1:
  sum_files = get_summary_file_name(x, contents2)
  sum_file_texts = {}
  for su_fl in sum_files:
    path = os.path.join(summary_path, su_fl)
    fl = open(path, 'r')
    txt = fl.read()
    sum_file_texts[su_fl] = txt
     
  path = os.path.join(annual_report_path, x)
  fl = open(path, 'r')
  txt = fl.read()
  name = x.split('.')[0]
  try:
    save_csv(txt, name, sum_file_texts, folder_path)
    print(len(contents1)-j)
    j += 1
  except:
    print("ERROR: "+x)
  
  
