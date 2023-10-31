from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mlxtend.preprocessing import DenseTransformer 
from feature_extractor import fasttext, fasttext2
from embedding_vectorizers import TfidfEmbeddingVectorizer, MeanEmbeddingVectorizer



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

    def transform(self, X, y=None):
    """ Return a slice of the input array. ... [rest of the docstring] ... """

    # We use the loc or iloc accessor if the input is a pandas dataframe
    if hasattr(X, 'loc') or hasattr(X, 'iloc'):
        # Ensure cols is a list
        if isinstance(self.cols, tuple):
            self.cols = list(self.cols)
        
        # Check all elements in `cols` are of the same data type
        types = {type(i) for i in self.cols}
        if len(types) > 1:
            raise ValueError('Elements in `cols` should be all of the same data type.')
        
        if isinstance(self.cols[0], int):
            t = X.iloc[:, self.cols].values
        elif isinstance(self.cols[0], str):
            t = X.loc[:, self.cols].values
        else:
            raise ValueError('Elements in `cols` should be either `int` or `str`.')
    else:
        t = X[:, self.cols]

    if t.shape[-1] == 1 and self.drop_axis:
        t = t.reshape(-1)
    elif len(t.shape) == 1 and not self.drop_axis:
        t = t[:, np.newaxis]
    
    return t



def create_pipeline(column, vectorizer, classifier):
    return Pipeline([
        ("col_sel", ColumnSelector(cols=column, drop_axis=True)),
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])

