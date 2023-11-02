import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# Base class for embedding vectorizers
class BaseEmbeddingVectorizer:
    # Initialization of the class with fasttext_embeddings
    def __init__(self, fasttext_embeddings):
        self.fasttext_embeddings = fasttext_embeddings  # Dictionary of fastText embeddings
        # Dimension of the embeddings, inferred from the first element if embeddings are provided
        self.dim = len(fasttext_embeddings[next(iter(fasttext_embeddings))]) if fasttext_embeddings else None

    # Fit method just returns itself, as no fitting is needed for the base vectorizer
    def fit(self, X, y=None):
        return self

    # Placeholder for the transform method, to be implemented by subclasses
    def transform(self, X):
        raise NotImplementedError()

    # Method to get the parameters of the vectorizer, for compatibility with scikit-learn's pipeline
    def get_params(self, deep=True):
        return {'fasttext_embeddings': self.fasttext_embeddings}

# Subclass that averages word embeddings to create document vectors
class MeanEmbeddingVectorizer(BaseEmbeddingVectorizer):
    # Transforms a list of tokenized texts (X) into a matrix of averaged word embeddings
    def transform(self, X):
        return np.array([
            # Compute the mean vector for each text, only including words that are in the embeddings
            np.mean([self.fasttext_embeddings[w] for w in words if w in self.fasttext_embeddings]
                    # If no words in the text are in the embeddings, use a zero vector of the same dimension
                    or [np.zeros(self.dim)], axis=0)
            for words in X  # X is a list of lists of tokens
        ])

# Subclass that weights word embeddings by their TF-IDF scores before averaging to create document vectors
class TfidfEmbeddingVectorizer(BaseEmbeddingVectorizer):
    # Fit method that initializes the TF-IDF vectorizer and calculates the word weights
    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer(analyzer=lambda x: x)  # Initialize TF-IDF with tokenized input
        self.tfidf.fit(X)  # Fit the TF-IDF vectorizer on the tokenized texts
        max_idf = max(self.tfidf.idf_)  # Maximum document frequency inverse
        # Default dict for word weights, defaulting to max_idf for words not in the vocabulary
        self.word2weight = defaultdict(lambda: max_idf,
                                       [(w, self.tfidf.idf_[i]) for w, i in self.tfidf.vocabulary_.items()])
        return self

    # Transform a list of tokenized texts (X) into a matrix of word embeddings weighted by TF-IDF
    def transform(self, X):
        return np.array([
            # Compute the weighted mean vector for each text
            np.mean([self.fasttext_embeddings[w] * self.word2weight[w] for w in words if w in self.fasttext_embeddings]
                    # If no words in the text are in the embeddings, use a zero vector
                    or [np.zeros(self.dim)], axis=0)
            for words in X  # X is a list of lists of tokens
        ])



 
