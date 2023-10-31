ffrom data_loader import process_data
from data_preprocessor import preprocess_data
from feature_extractor import extract_important_features, exclude_nouns, df_to_fasttext_data
from embedding_vectorizers import TfidfEmbeddingVectorizer, MeanEmbeddingVectorizer
from pipeline_constructors import create_pipeline, ColumnSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 0
TRAIN_SIZE = 0.9  
COL_INDEX = 7     



def build_classifier_pipelines():
    """Create and return classifier pipelines."""
    
    log_reg_fasttext_tfidf = Pipeline([
        ("col_sel", ColumnSelector(cols=COL_INDEX, drop_axis=True)),
        ("fasttext vectorizer", TfidfEmbeddingVectorizer(fasttext)),
        ("log_reg", LogisticRegression("l2", random_state=RANDOM_STATE))
    ])
    
    svm_fasttext = Pipeline([
        ("col_sel", ColumnSelector(cols=COL_INDEX, drop_axis=True)),
        ("fasttext vectorizer", MeanEmbeddingVectorizer(fasttext)),
        ("LinearSVC", SVC(random_state=RANDOM_STATE, kernel="linear", tol=1e-5, probability=True))
    ])
    
    # ... [other pipelines]
    
    return [svm_fasttext, svm_fasttext_tfidf, log_reg_fasttext_tfidf2]

def train_ensemble(X_train, y_train):
    """Train ensemble using given training data."""
    
    ensemble = SuperLearner(scorer=accuracy_score, random_state=RANDOM_STATE)
    ensemble.add(build_classifier_pipelines())
    ensemble.add_meta(LogisticRegression("l2", random_state=RANDOM_STATE))
    
    ensemble.fit(X_train, y_train)
    
    return ensemble


def main():
    base_path = 'processed_acl/'
    categories = ['books', 'dvd', 'kitchen', 'electronics']

    # Assuming that you always want to apply transformations or models on the same column, set a global COL_INDEX
    COL_INDEX = 7  # adjust as per your specific requirements

    for category in categories:
        df_list = []
        for sentiment in ['positive', 'negative']:
            df_list.append(process_data(base_path + category + '/', sentiment))

        # Assuming each df in df_list needs to be processed and trained on:
        for sample_df in df_list:
            sample_df = preprocess_data(sample_df)

            # Load fastText model (this can be outside the loop if you're using the same model for all categories)
            ft_model = load_model('Downloads/cc.en.300.bin')

            # Embedding and Model Training
            # Splitting the data
            X_train, X_test, y_train, y_test = train_test_split(
                sample_df.drop(columns=['label']).values, 
                sample_df.label.values,
                train_size=0.9, 
                random_state=0
            )
            
            # Train ensemble (or any other model/pipeline you'd like)
            ensemble_model = train_ensemble(X_train, y_train)
            
            # Predict on test set and print the accuracy
            preds = ensemble_model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            print(f"Accuracy for {category} - {sentiment}: {accuracy:.4f}")
