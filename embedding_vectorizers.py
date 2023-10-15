import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

class BaseEmbeddingVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec[next(iter(word2vec))]) if word2vec else None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError()

    def get_params(self, deep=True):
        return {'word2vec': self.word2vec}


class MeanEmbeddingVectorizer(BaseEmbeddingVectorizer):
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(BaseEmbeddingVectorizer):
    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer(analyzer=lambda x: x)
        self.tfidf.fit(X)
        max_idf = max(self.tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf,
                                       [(w, self.tfidf.idf_[i]) for w, i in self.tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# If you need the version with "fasttext2", just instantiate the above classes with different embeddings:
# mean_vectorizer = MeanEmbeddingVectorizer(fasttext2)
# tfidf_vectorizer = TfidfEmbeddingVectorizer(fasttext2)
 
