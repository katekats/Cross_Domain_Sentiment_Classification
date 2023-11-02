#from bs4 import BeautifulSoup
#import json
#import optparse
import os, regex as re
import pandas as pd

#import libraries 

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import re 
#import matplotlib.pyplot as plt 
#import seaborn as sns
import scipy as sp

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_score

from sklearn import naive_bayes
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from mlens.ensemble import SuperLearner

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

from nltk.stem.porter import PorterStemmer

import imblearn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler



def read_data(filename):
    list1 = []
    reviews = []
    with open(filename, 'r+') as fr:
        print(fr)
        end_of_review = 0
        for line in fr:
            #print("is not empty")
            line = re.sub(r'[:][\d]', " ", str(line))
            if (re.search("#label#:negative", str(line))):
                line = re.sub("#label#:negative", " " ,str(line))
                end_of_review=1
            if (re.search("#label#:positive", str(line))):
                line = re.sub("#label#:positive", " " ,str(line))
                end_of_review=1    
            str1 = str(line)    
                #print("end-of review")
            if end_of_review == 1:    
                reviews.append(str1)
                end_of_review = 0           
           
        return (reviews)
            #print(list1)      
        
def convert_to_dataframe(listname):
    df1 = pd.DataFrame({'reviews':listname})
    return df1

def get_label_from_filename(filename, df):
    if re.search("positive", str(filename)):
        df["label"] = 1
    if  re.search("negative", str(filename)):
        df["label"] = 0 
        #pd.set_option('display.max_colwidth', -1)
    return df   

# books_dataset
neg_reviews_list = read_data('Downloads/processed_acl/books/negative.review')
df1 = convert_to_dataframe(neg_reviews_list)
df1 =get_label_from_filename('Downloads/processed_acl/books/negative.review', df1)


pos_reviews_list = read_data('Downloads/processed_acl/books/positive.review')
df2 = convert_to_dataframe(pos_reviews_list)
df2 =get_label_from_filename('Downloads/processed_acl/books/positive.review', df2)


df_books = pd.concat([df1, df2], axis=0)

# dvd_dataset
neg_reviews_list2 = read_data('Downloads/processed_acl/dvd/negative.review')
df3 = convert_to_dataframe(neg_reviews_list2)
df3 =get_label_from_filename('Downloads/processed_acl/dvd/negative.review', df3)




pos_reviews_list2 = read_data('Downloads/processed_acl/dvd/positive.review')
df4 = convert_to_dataframe(pos_reviews_list2)
df4 =get_label_from_filename('Downloads/processed_acl/dvd/positive.review', df4)

df_dvd = pd.concat([df3, df4], axis=0)

# kitchen_dataset
neg_reviews_list3 = read_data('Downloads/processed_acl/kitchen/negative.review')
df5 = convert_to_dataframe(neg_reviews_list3)
df5 =get_label_from_filename('Downloads/processed_acl/kitchen/negative.review', df5)


pos_reviews_list3 = read_data('Downloads/processed_acl/kitchen/positive.review')
df6 = convert_to_dataframe(pos_reviews_list3)
df6 =get_label_from_filename('Downloads/processed_acl/kitchen/positive.review', df6)

df_kitchen = pd.concat([df5, df6], axis=0)

# electronics_dataset
neg_reviews_list4 = read_data('Downloads/processed_acl/electronics/negative.review')
df7 = convert_to_dataframe(neg_reviews_list4)
df7 =get_label_from_filename('Downloads/processed_acl/electronics/negative.review', df7)


pos_reviews_list4 = read_data('Downloads/processed_acl/electronics/positive.review')
df8 = convert_to_dataframe(pos_reviews_list4)
df8 =get_label_from_filename('Downloads/processed_acl/electronics/positive.review', df8)

df_electronics = pd.concat([df7, df8], axis=0)


#adding column for number of words in review in original data frame
df_books['#words'] = df_books.reviews.apply(lambda x: len(str(x).split(' ')))
df_dvd['#words'] = df_dvd.reviews.apply(lambda x: len(str(x).split(' ')))
#e_df['#words'] = e_df.reviewText.apply(lambda x: len(str(x).split(' ')))
#k_df['#words'] = k_df.reviewText.apply(lambda x: len(str(x).split(' ')))
df_kitchen['#words'] = df_kitchen.reviews.apply(lambda x: len(str(x).split(' ')))
df_electronics['#words'] = df_electronics.apply(lambda x: len(str(x).split(' ')))

#Shuffling the rows in all the datasets to make them randomly ordered
df_books.sample(frac=1)
df_books = df_books.sample(frac=1).reset_index(drop=True)
df_books["code"] = "books"

df_dvd.sample(frac=1)
df_dvd = df_dvd.sample(frac=1).reset_index(drop=True)
df_dvd["code"] = "dvd"

df_kitchen.sample(frac=1)
df_kitchen = df_kitchen.sample(frac=1).reset_index(drop=True)
df_kitchen["code"] = "kitchen"

df_electronics.sample(frac=1)
df_electronics = df_electronics.sample(frac=1).reset_index(drop=True)
df_electronics["code"] = "electronics"

#Appending the datasets CDSA 
bd = df_books.append(df_dvd, ignore_index=True)
bk = df_books.append(df_kitchen, ignore_index=True)
db = df_dvd.append(df_books, ignore_index=True)
eb = df_electronics.append(df_books, ignore_index=True)
kb = df_kitchen.append(df_books, ignore_index=True)
ed = df_electronics.append(df_dvd, ignore_index=True)
kd = df_kitchen.append(df_dvd, ignore_index=True)
be = df_books.append(df_electronics, ignore_index=True)
de = df_dvd.append(df_electronics, ignore_index=True)
ke = df_kitchen.append(df_electronics, ignore_index=True)
ek = df_electronics.append(df_kitchen, ignore_index=True)
dk = df_dvd.append(df_kitchen, ignore_index=True)



sample_df = df_dvd
#Functions for preprocessing steps
stop = set(('i','im','ive', 'me','my','myself','we','our','ours','ourselves','you','youre','youve','youll','youd','your','yours','yourself','yourselves','he','him','his','himself','she','shes','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','thatll','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','should','shouldve','now','d','ll','m','o','re','ve','y','ma'))
sno = nltk.stem.SnowballStemmer('english')

def replace_url(df,col,rm1,rm2):
    return(df[col].str.replace(rm1,rm2))

def extract_emo(df, col, emo):
    return(df[col].str.extractall(emo).unstack().apply(lambda x:' '.join(x.dropna()), axis=1))

def replace_emo(df,col,emo1,emo2):
    return(df[col].str.replace(emo1,emo2))

def replace_punct(df, col, punct1, punct2):
    return(df[col].str.replace(punct1, punct2))

def remove_numbers(df,col,rm1,rm2):
    return(df[col].str.replace(rm1,rm2))

def lower_words(df,col):
    return(df[col].apply(lambda x: " ".join(x.lower() for x in x.split())))

def remove_stop(df,col):
    return(df[col].apply(lambda x: " ".join(x for x in x.split() if x not in stop)))

def tokenize(df,col):
    return(df.apply(lambda row: nltk.word_tokenize(row[col]), axis=1))

def word_count(df,col):
    return(df[col].apply(lambda x: len(str(x).split(' '))))

def stemming(df,col):
    return(df[col].apply(lambda x: " ".join([sno.stem(word) for word in x.split()])))


#Step1 Pre-Processing
sample_df['nohtml'] = replace_url(sample_df,'reviews','^http?:\/\/.*[\r\n]*','')
sample_df['nohtml'] = lower_words(sample_df,'nohtml')
sample_df['nohtml'] = remove_numbers(sample_df, 'nohtml', '[0-9]+',' ')
sample_df['nohtml'] = replace_punct(sample_df, 'nohtml', '[^\w\s]',' ')
sample_df['nohtml'] = replace_punct(sample_df, 'nohtml', '_',' ')
sample_df['nohtml'] = replace_punct(sample_df, 'nohtml',r'\b(no|not|nt|dont|doesnt|doesn|don|didnt|cant|cannt|cannot|wouldnt|wont|couldnt|hasnt|havent|hadnt|shouldnt)\s+([a-z])',r'not \2')
sample_df['nohtml'] = remove_stop(sample_df,'nohtml')
#sample_df['nohtml'] = stemming(sample_df,'nohtml')
sample_df['tokenized'] = tokenize(sample_df,'nohtml')
sample_df['#token'] = word_count(sample_df,'tokenized')


sample_df_dvd =sample_df[sample_df["#token"]>75].reset_index(drop=True)
#sample_df_dvd =sample_df



sample_df_books=sample_df[sample_df["#token"]>80].reset_index(drop=True)
#sample_df_books =sample_df


sample_df_electronics=sample_df[sample_df["#token"]>=55].reset_index(drop=True)
#sample_df_electronics = sample_df



sample_df_kitchen=sample_df[sample_df["#token"]>=54].reset_index(drop=True)
#sample_df_kitchen = sample_df


#Appending the datasets CDSA 
bd = sample_df_books.append(sample_df_dvd, ignore_index=True)
bk = sample_df_books.append(sample_df_kitchen, ignore_index=True)
db = sample_df_dvd.append(sample_df_books, ignore_index=True)
eb = sample_df_electronics.append(sample_df_books, ignore_index=True)
kb = sample_df_kitchen.append(sample_df_books, ignore_index=True)
ed = sample_df_electronics.append(sample_df_dvd, ignore_index=True)
kd = sample_df_kitchen.append(sample_df_dvd, ignore_index=True)
be = sample_df_books.append(sample_df_electronics, ignore_index=True)
de = sample_df_dvd.append(sample_df_electronics, ignore_index=True)
ke = sample_df_kitchen.append(sample_df_electronics, ignore_index=True)
ek = sample_df_electronics.append(sample_df_kitchen, ignore_index=True)
dk = sample_df_dvd.append(sample_df_kitchen, ignore_index=True)
sample_df1=db



sample_df = sample_df1.copy()

# chi square for the important features per product category

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train1 = count_vect.fit_transform(sample_df.nohtml.values)
features1 = count_vect.get_feature_names()   
    
cat_chi2score0 = chi2(X_train1, sample_df.code)[0]
cat_chi2score1 = chi2(X_train1, sample_df.code)[1]
cat_wscores = zip(features1, cat_chi2score0)
cat_wchi2 = sorted(cat_wscores, key=lambda x:x[1])
#topchi2 = list(zip(*wchi2[-1000:]))
cat_topchi2score= cat_wchi2[-1000:]
#cat_chi2score0


#chi square for the important features per sentiment class

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(sample_df.nohtml[0:2000].values)
features = count_vect.get_feature_names()   
    
chi2score0 = chi2(X_train, sample_df.label[0:2000])[0]
chi2score1 = chi2(X_train, sample_df.label[0:2000])[1]
wscores = zip(features, chi2score0)
wchi2 = sorted(wscores, key=lambda x:x[1])
#topchi2 = list(zip(*wchi2[-1000:]))

topchi2score= wchi2[-6000:]
#topchi2score


# use only the important features
import collections

d4 = collections.OrderedDict((k, v) for k, v in zip(features1, cat_chi2score1) if v<0.05)
#print(d4)
list4 = [k for k, v in d4.items()]
d5 = collections.OrderedDict((k, v) for k, v in zip(features, chi2score1) if v<0.05)
list5 = [k for k, v in d5.items() if k not in d4.items()]
list6 = [k for k, v in d5.items()]
#d2 = collections.OrderedDict((k, v) for k, v in cat_chi2score)
d2 = collections.OrderedDict((k, v) for k, v in cat_topchi2score)
list3 = [k for k, v in d2.items()]

d = collections.OrderedDict((k, v) for k, v in topchi2score)
list1 = [k for k, v in d.items() if k not in d2.items()]
   
# keep the important features    
sample_df["tokenized1"] = sample_df.tokenized   
for  index, row in sample_df[0:1637].iterrows():
   
    row["tokenized1"] =  [word for word in row["tokenized1"] if word in list5]
    sample_df.at[index, 'tokenized1'] = row["tokenized1"]  
    
for  index2, row2 in sample_df[1637:1737].iterrows():
   
    row2["tokenized1"] =  [word for word in row2["tokenized1"] if word in list6]
    sample_df[1637:1737].set_value(index2,'tokenized1',row2["tokenized1"])  


# # Exclude nouns


sample_df["tokenized2"] = sample_df.tokenized   

noun = []
for  index, row in sample_df.iterrows():
    noun = [word for word,pos in pos_tag(row["tokenized2"]) if pos.startswith('N')]
    #print(noun)
    row["tokenized2"] =  [word for word in row["tokenized2"] if word not in noun]
    sample_df.set_value(index,'tokenized2',row["tokenized2"])


#from pyfasttext import FastText
#ft_model = FastText("Downloads/cc.en.300.bin")

from fastText import load_model
#import fastText 
#ft_model = fastText.load_model('Downloads/cc.en.300.bin')
#X1= sample_df.tokenized2
X=sample_df.tokenized1
X2=sample_df.tokenized2

#from fastText import load_model

#ft_model = load_model('Downloads/cc.en.300.bin')
#n_features = ft_model.get_dimension()
#dict1 ={}


def df_to_data(df, X):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    #x = np.zeros((len(df), 1000, n_features), dtype='float32')

    #for i, word in enumerate(sample_df['tokenized'].values):
    X=sample_df.tokenized
    all_words = set(w for words in X for w in words)
    for word in all_words:
            nums=ft_model.get_word_vector(word).astype('float32')
            dict1[word] = nums
            
     
    return dict1     

fasttext = df_to_data(sample_df, X)
fasttext2 = df_to_data(sample_df, X2)
# In[ ]:


#use the pretrained fasttext


#%matplotlib inline
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import StratifiedShuffleSplit
from collections import Counter, defaultdict


y= sample_df.label
import struct 


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, probs=True):
        self.word2vec = word2vec
        self.probs = probs
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(fasttext))])
    
    def get_params(self, deep=True):
        return dict(word2vec=self.word2vec)
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
   
    
  
    
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, probs=True):
        self.word2vec = word2vec
        self.probs = probs
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(fasttext))])
            
    def get_params(self, deep=True):
        return dict(word2vec=self.word2vec)        
        
    def fit(self, X,y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
    
    
    

            
            

class MeanEmbeddingVectorizer2(object):
    def __init__(self, word2vec, probs=True):
        self.word2vec = word2vec
        self.probs = probs
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(fasttext2))])
    
    def get_params(self, deep=True):
        return dict(word2vec=self.word2vec)
    
    def fit(self, X2, y):
        return self
            
 

    def transform(self, X2):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X1
        ])
    
    
     
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer2(object):
    def __init__(self, word2vec, probs=True):
        self.word2vec = word2vec
        self.probs = probs
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(fasttext2))])
            
    def get_params(self, deep=True):
        return dict(word2vec=self.word2vec)        
        
    def fit(self, X2,y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X2)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X2):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X2
            ])   

from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import make_pipeline
#from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from mlxtend.preprocessing import DenseTransformer 
    
from sklearn.base import BaseEstimator
import numpy as np


class ColumnSelector(BaseEstimator):
    """Object for selecting specific columns from a data set.
    Parameters
    ----------
    cols : array-like (default: None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.
    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/
    """

    def __init__(self, cols=None, drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

    def fit_transform(self, X, y=None):
        """ Return a slice of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features
        """
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        """ Return a slice of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features
        """

        # We use the loc or iloc accessor if the input is a pandas dataframe
        if hasattr(X, 'loc') or hasattr(X, 'iloc'):
            if type(self.cols) == tuple:
                self.cols = list(self.cols)
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError(
                    'Elements in `cols` should be all of the same data type.'
                )
            if isinstance(self.cols[0], int):
                t = X.iloc[:, self.cols].values
            elif isinstance(self.cols[0], str):
                t = X.loc[:, self.cols].values
            else:
                raise ValueError(
                    'Elements in `cols` should be either `int` or `str`.'
                )
        else:
            t = X[:, self.cols]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)
        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]
        return t

    def fit(self, X, y=None):
        """ Mock method. Does nothing.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        self
        """
        return self




log_reg_fasttext_tfidf = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer(fasttext)),
                        ("log_reg", LogisticRegression("l2", random_state=0))])

log_reg_fasttext2 = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", MeanEmbeddingVectorizer(fasttext)),
                        ("log_reg", LogisticRegression("l2", random_state=0))])

svm_fasttext = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", MeanEmbeddingVectorizer(fasttext)), 
                            ("LinearSVC", SVC(random_state=0, kernel="linear", tol=1e-5, probability=True))])

svm_fasttext_tfidf = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer(fasttext)), 
                            ("LinearSVC", SVC(random_state=0, kernel="linear", tol=1e-5, probability=True))])

log_reg_fasttext_tfidf2 = Pipeline([("col_sel", ColumnSelector(cols=8, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer2(fasttext2)),
                        ("log_reg", LogisticRegression("l2", random_state=0))])
#pipe_rf = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", MeanEmbeddingVectorizer2(fasttext2)),
#                        ('clf', RandomForestClassifier(n_estimators = 140, max_features = 60, max_depth =120,
#                                criterion = "gini",min_samples_split = 5, min_samples_leaf= 2,
#                                                       random_state=0))])

svm_fasttext_tfidf2 = Pipeline([("col_sel", ColumnSelector(cols=8, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer2(fasttext2)), 
                            ("LinearSVC", SVC(random_state=0, kernel="linear", tol=1e-5, probability=True))])


from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import make_pipeline
#from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from mlxtend.preprocessing import DenseTransformer 
    
from sklearn.base import BaseEstimator
import numpy as np


class ColumnSelector(BaseEstimator):
    """Object for selecting specific columns from a data set.
    Parameters
    ----------
    cols : array-like (default: None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.
    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/
    """

    def __init__(self, cols=None, drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

    def fit_transform(self, X, y=None):
        """ Return a slice of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features
        """
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        """ Return a slice of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features
        """

        # We use the loc or iloc accessor if the input is a pandas dataframe
        if hasattr(X, 'loc') or hasattr(X, 'iloc'):
            if type(self.cols) == tuple:
                self.cols = list(self.cols)
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError(
                    'Elements in `cols` should be all of the same data type.'
                )
            if isinstance(self.cols[0], int):
                t = X.iloc[:, self.cols].values
            elif isinstance(self.cols[0], str):
                t = X.loc[:, self.cols].values
            else:
                raise ValueError(
                    'Elements in `cols` should be either `int` or `str`.'
                )
        else:
            t = X[:, self.cols]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)
        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]
        return t

    def fit(self, X, y=None):
        """ Mock method. Does nothing.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        self
        """
        return self


#import mlxtend
#pipe1 = make_pipeline(ColumnSelector(cols=(7,)), MeanEmbeddingVectorizer2(fasttext2), LogisticRegression("l1", random_state=0))
#pipe2 = make_pipeline(ColumnSelector(cols=(5, )), TfidfEmbeddingVectorizer(fasttext), SVC(random_state=0, kernel="linear", tol=1e-5, probability=True))

#sclf = StackingClassifier(classifiers=[pipe1, pipe2], 
                         # meta_classifier=LogisticRegression())
# Fit ensemble
#sclf.fit(sample_df[0:1638], sample_df.label[0:1638].values)

# Predict
#preds = sclf.predict(sample_df[1639:])

#accuracy=accuracy_score(sample_df.label[1639:], preds)
#print(accuracy)

log_reg_fasttext_tfidf = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer(fasttext)),
                        ("log_reg", LogisticRegression("l2", random_state=0))])

log_reg_fasttext2 = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", MeanEmbeddingVectorizer(fasttext)),
                        ("log_reg", LogisticRegression("l2", random_state=0))])

svm_fasttext = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", MeanEmbeddingVectorizer(fasttext)), 
                            ("LinearSVC", SVC(random_state=0, kernel="linear", tol=1e-5, probability=True))])


log_reg_fasttext_tfidf2 = Pipeline([("col_sel", ColumnSelector(cols=8, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer2(fasttext2)),
                        ("log_reg", LogisticRegression("l2", random_state=0))])
#pipe_rf = Pipeline([("col_sel", ColumnSelector(cols=7, drop_axis=True)), ("fasttext vectorizer", MeanEmbeddingVectorizer2(fasttext2)),
#                        ('clf', RandomForestClassifier(n_estimators = 140, max_features = 60, max_depth =120,
#                                criterion = "gini",min_samples_split = 5, min_samples_leaf= 2,
#                                                       random_state=0))])

svm_fasttext_tfidf = Pipeline([("col_sel", ColumnSelector(cols=8, drop_axis=True)), ("fasttext vectorizer", TfidfEmbeddingVectorizer2(fasttext2)), 
                            ("LinearSVC", SVC(random_state=0, kernel="linear", tol=1e-5, probability=True))])


# In[ ]:


from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import make_pipeline
seed = 0
#np.random.seed(seed)
ensemble = SuperLearner(scorer=metrics.accuracy_score, random_state=seed)

# Build the first -rflayer
ensemble.add([svm_fasttext, svm_fasttext_tfidf, log_reg_fasttext_tfidf2] )

# Attach the final meta estimator
ensemble.add_meta(LogisticRegression("l2", random_state=0))
# --- Use ---

# Fit ensemble
ensemble.fit(sample_df[0:1737].values, sample_df.label[0:1737].values)

# Predict
preds = ensemble.predict(sample_df[1737:].values)

accuracy=accuracy_score(sample_df.label[1737:].values, preds)
print(accuracy)

