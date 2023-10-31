from data_loader import process_data
from data_preprocessor import preprocess_data
from feature_extractor import extract_important_features, exclude_nouns, df_to_fasttext_data
from embedding_vectorizers import TfidfEmbeddingVectorizer, MeanEmbeddingVectorizer
from pipeline_constructors import create_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlxtend.classifier import SuperLearner
import fasttext  # assuming you have installed the fasttext library
from feature_extractor_var import sample_df, fasttext, fasttext2
from embedding_vectorizers import TfidfEmbeddingVectorizer, MeanEmbeddingVectorizer
import data_loader 
import data_preprocessor
from feature_extractor import main as feature_extractor_main

# define constraints
COL_INDEX=7


def build_preprocessing_pipeline():
    """Create and return preprocessing pipeline without classifier."""
    # Assuming COL_INDEX and MeanEmbeddingVectorizer(fasttext) are the preprocessing steps
    preprocessing_pipeline = Pipeline([
        ('selector', ColumnSelector(columns=COL_INDEX)),
        ('mean_vectorizer', MeanEmbeddingVectorizer(fasttext))
    ])
    return preprocessing_pipeline

def build_log_reg_fasttext_pipeline():
    """Create and return logistic regression with fasttext pipeline."""
    logistic_l2 = LogisticRegression(penalty="l2", random_state=0)
    return create_pipeline(COL_INDEX, MeanEmbeddingVectorizer(fasttext), logistic_l2)

def build_svm_fasttext_pipeline():
    """Create and return SVM with fasttext pipeline."""
    svc_linear = SVC(random_state=0, kernel="linear", tol=1e-5, probability=True)
    return create_pipeline(COL_INDEX, MeanEmbeddingVectorizer(fasttext), svc_linear)

def train_ensemble(X_train, y_train):
    """Train ensemble using given training data."""
    
    ensemble = SuperLearner(scorer=accuracy_score, random_state=RANDOM_STATE)
    ensemble.add(build_classifier_pipelines())
    
    # Adding the desired pipelines and classifiers to meta layer
    meta_models = [
        build_preprocessing_pipeline(),
        LogisticRegression(penalty="l2", random_state=RANDOM_STATE),
        build_log_reg_fasttext_pipeline(),
        build_svm_fasttext_pipeline()
    ]
    ensemble.add_meta(meta_models)
    
    ensemble.fit(X_train, y_train)
    
    return ensemble


def main():
    df_books, df_dvd, df_kitchen, df_electronics = data_loader.main()
    datasets, combinations = data_preprocessor.preprocess_datasets(df_books, df_dvd, df_kitchen, df_electronics)
    
    all_dataset_keys = ['bd', 'bk', 'db', 'eb', 'kb', 'ed', 'kd', 'be', 'de', 'ke', 'dk']
    
    results = []  # to store accuracy results for each dataset_key
    
    for dataset_key in all_dataset_keys:
        print(f"Processing for dataset key: {dataset_key}")
        
        sample_df, fasttext, desired_last_row_number = feature_extractor_main(None, datasets, combinations, dataset_key)
        
        # Training data
        X_train = sample_df.iloc[:desired_last_row_number + 101].drop(columns=['label']).values
        y_train = sample_df.iloc[:desired_last_row_number + 101].label.values
        
        # Randomly sample 100 rows for testing from the data after desired_last_row_number + 101
        test_data = sample_df.iloc[desired_last_row_number + 101:].sample(n=100, random_state=RANDOM_STATE)
        
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data.label.values
        
        ensemble_model = train_ensemble(X_train, y_train)
        
        preds = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results.append((dataset_key, accuracy))

    # Convert results to DataFrame for better visualization
    df_results = pd.DataFrame(results, columns=["Dataset Key", "Accuracy"])
    print(df_results)
