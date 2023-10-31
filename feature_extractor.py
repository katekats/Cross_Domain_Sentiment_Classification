import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from nltk.tag import pos_tag
from fastText import load_model
import numpy as np
import os
import urllib.request
import argparse
from data_preprocessor import datasets, combinations

def compute_chi2(data, labels, feature_limit=None):
    """
    Computes the chi-squared statistics for given data and labels.
    
    Parameters:
    - data: The textual data to be transformed and analyzed.
    - labels: The corresponding labels for the data.
    - feature_limit: Number of top features to return based on chi2 scores. If None, returns all features.
    
    Returns:
    - A tuple containing:
        1. The top chi2 scored features (up to the specified limit) along with their scores.
        2. A list of all feature names.
        3. chi2 p-values for each feature.
    """
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(data)
    features = count_vect.get_feature_names()   
    chi2score0 = chi2(X, labels)[0]
    chi2score1 = chi2(X, labels)[1]
    wscores = zip(features, chi2score0)
    sorted_wchi2 = sorted(wscores, key=lambda x:x[1])
    return sorted_wchi2[-feature_limit:] if feature_limit else sorted_wchi2, features, chi2score1

def extract_features_below_threshold(features, scores, threshold):
     """
    Filters the features based on a specified threshold.
    
    Parameters:
    - features: A list of feature names.
    - scores: A list of scores corresponding to each feature.
    - threshold: The score threshold below which features will be considered.
    
    Returns:
    - A list of features that have scores below the given threshold.
    """
    return [k for k, v in zip(features, scores) if v < threshold]

def modify_tokenized_data(sample_df, list_sent_gen, list_sent_domain):
      """
    Modifies the 'tokenized1' column of the input dataframe based on specified feature lists.
    
    Parameters:
    - sample_df: The input dataframe.
    - list_sent_gen: A list of general sentiment features.
    - list_sent_domain: A list of domain-specific sentiment features.
    
    Returns:
    - Modified dataframe with updated 'tokenized1' column.
    """
    sample_df["tokenized1"] = sample_df.tokenized   
    for index, row in sample_df[0:1637].iterrows():
        row["tokenized1"] = [word for word in row["tokenized1"] if word in list_sent_gen]
        sample_df.at[index, 'tokenized1'] = row["tokenized1"]  
    
    for index2, row2 in sample_df[1637:1737].iterrows():
        row2["tokenized1"] = [word for word in row2["tokenized1"] if word in list_sent_domain]
        sample_df.at[index2, 'tokenized1'] = row2["tokenized1"]

    return sample_df

def exclude_nouns(tokenized_data):
    return [[word for word in words if not word.startswith('N')] for words in tokenized_data]

def df_to_data(model, df, tokenized_column):
    all_words = set(w for words in df[tokenized_column] for w in words)
    return {word: model.get_word_vector(word).astype('float32') for word in all_words}


def download_fasttext_model(model_name="cc.en.300.bin"):
    """Download the specified FastText pre-trained model."""
    
    base_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
    download_link = os.path.join(base_url, model_name + ".gz")
    output_path = model_name + ".gz"

    # Check if file already exists
    if not os.path.exists(output_path):
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(download_link, output_path)
        print(f"Download complete: {output_path}")
    else:
        print(f"{model_name} already exists.")
    
    # Decompress the file (assuming it's a gzipped file)
    os.system(f"gunzip -k {output_path}")

    return model_name



def main(sample_df):
    # Assuming you have a function to load or define datasets and combinations
    
    combined_datasets = {
        key: datasets[df1].append(datasets[df2], ignore_index=True) 
        for key, (df1, df2) in combinations.items()
    }
    
    sample_df = combined_datasets.get(args.dataset_key).copy()

    # Compute the chi-squared statistics for important features with respect to the product domain.
    # 'cat_topchi2score' will store the top features and their corresponding chi2 scores.
    # 'features1' is a list of feature names and 'cat_chi2score1' contains the p-values for each feature.
    cat_topchi2score, features1, cat_chi2score1 = compute_chi2(sample_df.nohtml.values, sample_df.code, 1000)
    # Compute the chi-squared statistics for important features with respect to sentiment classification.
    # 'topchi2score' will store the top features and their corresponding chi2 scores for sentiment.
    # 'features' is a list of feature names and 'chi2score1' contains the p-values for each feature.
    topchi2score, features, chi2score1 = compute_chi2(sample_df.nohtml[0:2000].values, sample_df.label[0:2000], 6000)

    # 'list_domain' contains features significant for product domain classification.
    # It filters features with p-values below the threshold of 0.05, which means these features are statistically significant.
    list_domain = extract_features_below_threshold(features1, cat_chi2score1, 0.05)
    # 'list_sentiment' contains features significant for sentiment classification.
    # It filters features with p-values below the threshold of 0.05.
    list_sentiment = extract_features_below_threshold(features, chi2score1, 0.05)
     # 'list_sent_gen' contains the sentiment features that are not present in domain features.
    # This step is to ensure that general sentiment features that are not domain-specific are identified.
    list_sent_gen = [k for k in list_sentiment if k not in list_domain]
    # 'list_sent_domain' contains all sentiment features.
    # This list captures all the features considered during chi2 calculations for sentiment classification.
    list_sent_domain = [k for k, v in zip(features, chi2score1)]
    
    # Modifying tokenized data
    sample_df = modify_tokenized_data(sample_df, list_sent_gen, list_sent_domain)

    # Exclude nouns
    sample_df["tokenized2"] = exclude_nouns(sample_df["tokenized"])

    # Convert to fastText embeddings
    ft_model = download_fasttext_model()
    fasttext = df_to_data(ft_model, sample_df, 'tokenized')
    fasttext2 = df_to_data(ft_model, sample_df, 'tokenized2')

    return sample_df, fasttext, fasttext2

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Feature Extractor for Datasets.')
    parser.add_argument('--dataset_key', type=str, required=True,
                        help='Key for the dataset to be processed, e.g., "db"')
    args = parser.parse_args()
    
    sample_df, fasttext, fasttext2 = main(args)


