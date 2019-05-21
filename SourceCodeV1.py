# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:48:34 2019

@author: vinay
"""

#importing required packages

import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import re, collections
from collections import defaultdict
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from __future__ import division
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# load the learning data

data = pd.read_csv('C:\\Users\\vinay\\Desktop\\NLP\\Automated-Essay-Scoring-master\\Automated-Essay-Scoring-master\\essays_and_scores.csv', encoding = 'latin-1')
data=data[['essay_set','essay','domain1_score']]

# Tokenize a sentence into words

def sentence_to_wordlist(raw_sentence):
    clean_sentence = re.sub(r'\W', ' ', raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    return tokens

# tokenizing an essay into a list of word lists

def tokenize(essay):
    stripped_essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    return tokenized_sentences

# calculating number of characters in an essay

def character_count(essay):
    character = re.sub(r'\s', '', str(essay.encode('utf-8')).lower())
    return len(character)

# calculating number of words in an essay

def word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)  
    return len(words)

# calculating number of sentences in an essay

def sentence_count(essay):
    sentences = nltk.sent_tokenize(essay)
    return len(sentences)

# calculating average word length in an essay

def avg_word_len(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    return sum(len(word) for word in words) / len(words)

# long word count

def complex_word_count(essay):
    complex_word_count = 0
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    for word in words:
        if(len(word)>7):
            complex_word_count += 1
    return complex_word_count


# calculating number of lemmas per essay

def lemmas_count(essay):   
    tokenized_sentences = tokenize(essay)        
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence) 
        for token_tuple in tagged_tokens:  
            #extracting the tag from the word-tag tuple
            pos_tag = token_tuple[1]
            #lemmatize() accepts only a list of pos-tags hence the tags need to be manipulated
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))   
    lemma_count = len(set(lemmas))
    return lemma_count


#read the dictionary file
data = open('C:\\Users\\vinay\\Desktop\\NLP\\Automated-Essay-Scoring-master\\Automated-Essay-Scoring-master\\big.txt').read()
#extract all words from the dictionary
words_ = re.findall('[a-z]+', data.lower())
#generate dictionary template
word_dict = collections.defaultdict(lambda: 0)
#fill dictionary template                   
for word in words_:
    word_dict[word] += 1
d = word_dict.items()
dictionary = pd.DataFrame(d)
dictionary.columns = ['Word','Value']
dictionary['Length'] = dictionary['Word'].apply(len)
# checking number of misspelled words

def spelling_error_count(essay):   
    clean_essay = re.sub(r'\W', ' ', str(essay.encode('utf-8')).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    #mis-spelled word counter initialization      
    mispell_count = 0
    #words from essay
    words = clean_essay.split()                    
    for word in words:
        d=dictionary[dictionary['Length']==len(word)]
        if not word in d['Word']:
            mispell_count += 1
    return mispell_count

# calculating number of nouns, adjectives, verbs and adverbs in an essay

def pos_count(essay):
    tokenized_sentences = tokenize(essay)
    #counter initialization
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    proper_noun_count = 0
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        for token_tuple in tagged_tokens:
            #extracting the tag from the word-tag tuple
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            elif (pos_tag == 'NNP'):
                proper_noun_count +=1
    return noun_count, proper_noun_count, adj_count, verb_count, adv_count

#BOW
    
def get_count_vectors(essays):
    vectorizer = CountVectorizer(max_features = 100, ngram_range=(1, 3), stop_words='english')
    count_vectors = vectorizer.fit_transform(essays)    
    feature_names = vectorizer.get_feature_names()    
    return feature_names, count_vectors

# extracting essay features

def extract_features(data):
    features = data.copy()
    features['character_count'] = features['essay'].apply(character_count)
    features['word_count'] = features['essay'].apply(word_count)
    features['sentence_count'] = features['essay'].apply(sentence_count)
    features['avg_word_len'] = features['essay'].apply(avg_word_len)
    features['lemmas_count'] = features['essay'].apply(lemmas_count)
    features['spelling_err_count'] = features['essay'].apply(spelling_error_count) 
    features['noun_count'],features['proper_noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(pos_count))
    features['complex_word_count'] = features['essay'].apply(complex_word_count)
    return features

#Delta

def Delta(y_pred,y_test,c):
    p1=pd.DataFrame(y_pred)
    p1=p1.reset_index()
    p2=pd.DataFrame(y_test)
    p2=p2.reset_index()

    p1.columns = ['id','Prediction']
    p2.columns = ['id','True Value']

    P=pd.merge(p1,p2,on='id')
    if (c==1):
        P['Predictions Rounded'] = P['Prediction'].apply(round)
        P['Delta_R'] = P['True Value'] -P['Predictions Rounded']
        P['Zero'] = (P['Delta_R']==0)
    else:
        P['Delta'] = P['True Value'] - P['Prediction']
        P['Zero'] = (P['Delta']==0)
    
    return sum(P['Zero'])

#Display

def display(y_test, y_pred, c):
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: %.4f" % mse)
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % grid.score(X_test, y_test))
    # Cohen’s kappa score: 1 is complete agreement
    print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
    delta = Delta(y_pred,y_test,c)
    d = y_pred.shape[0]
    dp = (delta*100)/d
    print('Delta score: %.2f' %dp)

#Gradient Boosting Regressor
    
def GBR(X_train, X_test, y_train, y_test):
    params = {'n_estimators':[50, 100, 500, 1000], 'max_depth':[2], 'min_samples_split': [2],
          'learning_rate':[1, 0.1, 0.3, 0.01], 'loss': ['ls']}
    gbr = ensemble.GradientBoostingRegressor()
    grid = GridSearchCV(gbr, params)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    #print(grid.best_score_)
    #print(grid.best_estimator_)
    print("\nClassifier: Gradient Boosting Regressor")
    display(y_test, y_pred, 1)

#Decision Tree Classifier
    
def DTC(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    fit = clf.fit(X_train, y_train)
    y_pred = fit.predict(X_test)
    print("\nClassifier: Descision Tree Classifier")
    display(y_test, y_pred, 0)

#Logistic Regression

def LogR(X_train, X_test, y_train, y_test):
    lr= LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("\nClassifier: Logistic Regression")
    display(y_test, y_pred, 1)

#Linear Regression

def LR(X_train, X_test, y_train, y_test):
    lr=LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("\nClassifier: Linear Regression")
    display(y_test, y_pred, 1)

#Random Forest

def RF(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nClassifier: Random Forest")
    display(y_test, y_pred, 0)

#Naive Bayes

def NB(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("\nClassifier: Gaussian Naive Bayes")
    display(y_test, y_pred, 0)

#Classifiers
    
def classifiers(X_train, X_test, y_train, y_test):
    GBR(X_train, X_test, y_train, y_test)
    DTC(X_train, X_test, y_train, y_test)
    LogR(X_train, X_test, y_train, y_test)
    LR(X_train, X_test, y_train, y_test)
    RF(X_train, X_test, y_train, y_test)
    NB(X_train, X_test, y_train, y_test)   

# extracting features from essay set 1
data_set1 = data[data['essay_set'] == 1]
essays = data_set1['essay']
features_set1 = extract_features(data_set1)
feature_names_cv, count_vectors = get_count_vectors(essays)
X_cv = count_vectors.toarray()
X = features_set1.iloc[:, 3:].as_matrix()
X = np.concatenate((features_set1.iloc[:, 3:].as_matrix(), X_cv), axis = 1)
y = features_set1['domain1_score'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
classifiers(X_train, X_test, y_train, y_test)

plt.scatter(features_set1['domain1_score'], features_set1['character_count'])
plt.xlabel('Score')
plt.ylabel('Character Count')
plt.show()

plt.scatter(features_set1['domain1_score'], features_set1['word_count'])
plt.xlabel('Score')
plt.ylabel('Word Count')
plt.show()