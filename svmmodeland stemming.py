#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
from sklearn import svm
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


# In[2]:


IMDB = pd.read_csv('imdb_labelled.txt', sep="\t", quoting=3, header=None)
AMAZON= pd.read_csv('amazon_cells_labelled.txt', sep="\t", header=None)
YELP= pd.read_csv('yelp_labelled.txt', sep="\t", header=None)


# In[3]:


#3.1
from nltk.tokenize import word_tokenize


# In[4]:


frames = [IMDB, AMAZON, YELP]
result = pd.concat(frames, ignore_index=True)
# χωρίζουμε sentences and scores
sentences=result.loc[:,0]
scores=result.loc[:,1]


# In[5]:


# εξαγωγή λέξεων για να δημιουργήσουμε τη λίστα 
words=[]
words_temp=[]
for i in range(0, len(sentences)):   
    words_of_sentence=word_tokenize(sentences[i])
    words_temp.append(words_of_sentence)

words=[item for sublist in words_temp for item in sublist]
# stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
# αφαίρεση stop words
stop_words = set(stopwords.words('english'))
words = [word for word in stemmed_words if not word in stop_words]
#αφαίρεση των σημείων στίξης
words = [''.join(c for c in s if c not in string.punctuation) for s in words]
# αφαίρεση των κενών
words =  [word for word in words if word]
# αφαιρούμε τις επαναλήψεις για να υπάρχει κάθε λέξη μία φορά 
words = list(set(words))
# μετατρέπουμε τα κεφαλαία γράμματα σε μικρά 
words = [w.lower() for w in words]
# κάνουμε sort τη λίστα(ταξινόμηση κατά αύξουσα σειρά)
words.sort()
# εμφανίζουμε το συνολικό αριθμό των λέξεων
len(words)


# In[6]:


#3.2
# φτιάχνουμε ένα σύνολο δεδομένων όπου κάθε πρόταση πλεόν αναπαρίσταται με ένα διάνυσμα με ενα label
sentence_vector=np.zeros((len(sentences), len(words)), dtype=int)

for s in range(0, len(sentences)):
# μετατρέπουμε τα κεφαλαία γράμματα σε μικρά 
    sentence = sentences[s].lower()
 # αφαιρούμε τα στοχεία στίξης α΄πό τις προτάσεις
    sentence="".join(l for l in sentence if l not in string.punctuation)
# εξάγουμε τις λέξεις
    words_of_sentence=word_tokenize(sentence)
# stemming
    words_of_sentence_stemmed=[stemmer.stem(word) for word in words_of_sentence]
# αφαιρούμε τισ  stop words
    words_of_sentence_filtered = [word for word in words_of_sentence_stemmed if not word in stop_words]
    
    
# μετατρέπουμε τη λίστα με τις λέξεις σε διάνυσμα
    for i in range(0, len(words_of_sentence_filtered)):
        for j in range(0, len(words)):
            if (words_of_sentence_filtered[i]==words[j]):
                sentence_vector[s,j]=sentence_vector[s,j]+1


# In[7]:


print(sentence_vector)


# In[8]:


#ελεγχος
sentence_vector[250]


# In[9]:


sentences[25]


# In[10]:


#3.3
# θέτουμε για χ τα διανύσματα των προτάσεων και για y τα  scores
x_train, x_test, y_train, y_test = train_test_split(sentence_vector, scores, test_size=0.25)
#κατασκεύη του μοντέλου 
model=svm.SVC(kernel='linear', decision_function_shape='ovo')
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test, y_pred)













# In[11]:


#επιλογή των υπερπαραμέτρων
C_range=np.array([x * 0.1 for x in range(2, 10)])
C_scores=[]
for c in C_range:
    model=svm.SVC(C=c, kernel='linear', decision_function_shape='ovo')
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    C_scores.append(scores.mean())
print(C_scores)


# In[12]:


print(c)

# In[13]:


#διάγραμμα
plt.plot(C_range, C_scores)
plt.xlabel("Value C for SVM")
plt.ylabel("Cross-validated Accuracy")

# In[18]:


#τελικό μοντέλο 
model=svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovo' )
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test, y_pred)


# In[ ]:




