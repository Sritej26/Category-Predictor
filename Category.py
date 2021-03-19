# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:22:20 2021

@author: Sritej. N
"""
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

data= pd.read_csv("Bookcategory.csv", na_filter=True, na_values='[]')
data.dropna()
df=data[["title","subjects"]] #taking only titles and categories column and dropping the rest of info
#df=df.iloc[1:]


df=df.dropna(subset=['title'])
df[df['subjects'].astype(bool)] 


df=df[df.astype(str)['subjects'] != 'set()']
df[df.astype(str)['subjects'] != '[]']


from ast import literal_eval
def f(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        print(e)
        return []

#converting the data type of elements in subject column from string to list

df['subjects'] = df.subjects.apply(lambda x: f(x))   

v={}
for j in df['subjects'].values:    
    for i in j:
        if i not in v:          #calculating frequency distribution of categories
            v[i]=1
        else:
            v[i]+=1
    
#taking top 100 categories
v=sorted(v.items(), key=lambda item: item[1], reverse=True)
v=v[:100]

t=[] #for titles that are categorized into one of top 100 subjects
s=[] # for 100 categories with most frequency

for i in v:
    for index, row in df.iterrows():
        if i[0] in row['subjects']:
            t.append(row['title'])
            temp=[]
            temp.append(i[0])
            s.append(temp)
#creating new dataframe with 100 most frequent subjects and their corresponding book title
df1=pd.DataFrame()           
#list_of_tuples = list(zip(t,s))  
df1['title']=t
df1['subjects']=s



multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df1['subjects'])

tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,3), stop_words='english')
X = tfidf.fit_transform(df1['title'])
#included parameter stratify to equally split the labels among train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)




sgd = SGDClassifier()
lr = LogisticRegression(solver='lbfgs')
svc = LinearSVC()

# for testing the accuracy
def j_score(y_true, y_pred):
  jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
  return jaccard.mean()*100


def print_score(y_pred, clf):
  print("Clf: ", clf.__class__.__name__)
  print('Jacard score: {}'.format(j_score(y_test, y_pred)))
  print('----')


# using 3 diff classifiers to see their score
for classifier in [sgd, lr, svc]:
  clf = OneVsRestClassifier(classifier)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(y_pred, classifier)


# example case to test
x = [ 'red']
xt = tfidf.transform(x)
clf.predict(xt)
multilabel.inverse_transform(clf.predict(xt))













