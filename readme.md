# Cross-Domain Sentiment Classification with Chi-square for Feature Extraction and Ensembling

This project focuses on text classification using various feature extraction techniques, including TF-IDF, FastText embeddings, and Chi-squared statistics. Additionally, an ensemble learning approach using SuperLearner from mlxtend is employed.

## Getting Started
For running the framework, this work recommends creating a new virtual environment that uses Python version 3.7.7.
Afterward, you can install the packages in the requirements.txt of the requirements_files directory to get started. Using anaconda, the commands look like this:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## **Dataset**
The dataset for Cross-Domain Sentiment Classification can be found in the following link:
https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

## **Steps for running the scripts**

**For the CRD-SentEnse Model:**

```
├── data_loader.py          # Script to load and process raw data
├── data_preprocessor.py    # Script containing data preprocessing functions
├── feature_extractor.py    # Script for feature extraction and  fasttext embeddings. It contains functions like extract_important_features for Chi-squared feature extraction, exclude_nouns to filter out noun words, and df_to_fasttext_data for generating FastText embeddings.
├── embedding_vectorizers.py # Script with classes for averaging word embeddings and for aggregating embeddings with tfidf weights
├── pipeline_constructors.py # Script that creates the model's pipelines
├──model_training.py         # Script for training the model
├──Statistical_test_nouns.py  # Script for identifying the statistical importance of nouns and other part-of-speech (postags).

```
For the __CRD-SentEnse Var model__, we need to replace the __feature_extractor.py__ with __feature_extractor_var.py__ and __model_training.py__ with  __model_training_var.py__.

For running the Cross-Domain Sentiment Classification Model (CRD-SentEnse) we need only to run  __model_tainning.py__, while for running the variation CRD-SentEnse-var we should run the __model_tainning_var.py__.

The possible combinations for source and target domains are:

 combinations = {
        'bd': ('books', 'dvd'),
        'bk': ('books', 'kitchen'),
        'db': ('dvd', 'books'),
        'eb': ('electronics', 'books'),
        'kb': ('kitchen', 'books'),
        'ed': ('electronics', 'dvd'),
        'kd': ('kitchen', 'dvd'),
        'be': ('books', 'electronics'),
        'de': ('dvd','electronics'),
        'ke': ('kitchen', 'electronics'),
        'ek': ('electronics', 'kitchen'),
        'dk': ('dvd','kitchen')}
        
.


This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE.
