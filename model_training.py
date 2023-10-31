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
from feature_extractor import sample_df, fasttext, fasttext2
from embedding_vectorizers import TfidfEmbeddingVectorizer, MeanEmbeddingVectorizer
import data_loader 
import data_preprocessor
from feature_extractor import main as feature_extractor_main


RANDOM_STATE = 0
TRAIN_SIZE = 0.9  
COL_INDEX = 7  # adjust as per your specific requirements

def build_classifier_pipelines():
    """Create and return classifier pipelines."""
    
    logistic_l2 = LogisticRegression(penalty="l2", random_state=0)
    svc_linear = SVC(random_state=0, kernel="linear", tol=1e-5, probability=True)

    # Construct pipelines
    log_reg_fasttext_tfidf = create_pipeline(COL_INDEX, TfidfEmbeddingVectorizer(fasttext), logistic_l2)
    log_reg_fasttext = create_pipeline(COL_INDEX, MeanEmbeddingVectorizer(fasttext), logistic_l2)
    svm_fasttext = create_pipeline(COL_INDEX, MeanEmbeddingVectorizer(fasttext), svc_linear)
    svm_fasttext_tfidf = create_pipeline(COL_INDEX, TfidfEmbeddingVectorizer(fasttext), svc_linear)
    log_reg_fasttext_tfidf2 = create_pipeline(COL_INDEX+1, TfidfEmbeddingVectorizer(fasttext2), logistic_l2)
    svm_fasttext_tfidf2 = create_pipeline(COL_INDEX+1, TfidfEmbeddingVectorizer(fasttext2), svc_linear)
       
    return [log_reg_fasttext_tfidf, log_reg_fasttext, svm_fasttext, svm_fasttext_tfidf, log_reg_fasttext_tfidf2]

def train_ensemble(X_train, y_train):
    """Train ensemble using given training data."""
    
    ensemble = SuperLearner(scorer=accuracy_score, random_state=RANDOM_STATE)
    ensemble.add(build_classifier_pipelines())
    ensemble.add_meta(LogisticRegression(penalty="l2", random_state=RANDOM_STATE))
    
    ensemble.fit(X_train, y_train)
    
    return ensemble

def main():
    df_books, df_dvd, df_kitchen, df_electronics = data_loader.main()
    datasets, combinations = data_preprocessor.preprocess_datasets(df_books, df_dvd, df_kitchen, df_electronics)
    
    all_dataset_keys = ['bd', 'bk', 'db', 'eb', 'kb', 'ed', 'kd', 'be', 'de', 'ke', 'dk']
    
    results = []  # to store accuracy results for each dataset_key
    
    for dataset_key in all_dataset_keys:
        print(f"Processing for dataset key: {dataset_key}")
        
        sample_df, fasttext, fasttext2 = feature_extractor_main(None, datasets, combinations, dataset_key)
        
        X_train, X_test, y_train, y_test = train_test_split(
            sample_df.drop(columns=['label']).values, 
            sample_df.label.values,
            train_size=TRAIN_SIZE, 
            random_state=RANDOM_STATE
        )
        
        ensemble_model = train_ensemble(X_train, y_train)
        
        preds = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results.append((dataset_key, accuracy))

    # Convert results to DataFrame for better visualization
    df_results = pd.DataFrame(results, columns=["Dataset Key", "Accuracy"])
    print(df_results)

if __name__ == "__main__":
    main()