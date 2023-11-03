# Cross-Domain Sentiment Classification with Chi-square for Feature Extraction and Ensembling

This project focuses on text classification using various feature extraction techniques, including TF-IDF, FastText embeddings, and Chi-squared statistics. Additionally, an ensemble learning approach using SuperLearner from mlxtend is employed. The code is part of the paper: "Katerina Katsarou and Devvrat Singh Shekhawat. 2020. CRD-SentEnse: Cross-domain Sentiment Analysis using an Ensemble Model. In Proceedings of the 11th International Conference on Management of Digital EcoSystems (MEDES '19). Association for Computing Machinery, New York, NY, USA, 88–94. https://doi.org/10.1145/3297662.3365808".
Micro-blogging and comments on social media include valuable information about people's emotions and opinions towards products, political and social topics and so forth. Unfortunately, due to the large volume of data, is infeasible to label all these comments and reviews. Additionally, having this data labelled manually by human experts is very expensive, time-consuming and applicable only for small amounts of data. As a result, a more scalable solution is needed. Cross-domain sentiment analysis addresses the problem of training a model for classifying a text with respect to its sentiment polarity as a negative, positive (and/ or neutral), using data from one domain (source domain), then the same model is tested using data from a different unlabeled domain (target domain). Cross-domain sentiment analysis is still an open research issue, as the classification performance is still not as good as in the in-domain sentiment analysis, even though proposed approaches have improved significantly. In this paper, we propose a framework for cross-domain sentiment analysis that uses the chi-square test with the data in the source domain. Firstly, we eliminate domain-related words from the source domain that do not bear transferable knowledge to the target domain. Secondly, the chi-square test is utilized for finding the important words regarding the sentiment polarity. Subsequently, we develop a second model that drops the nouns both from source and target domains and we use TFIDF weights for finding the important words in both domains. Finally, we use a stacking ensemble model that combines the two above proposed models for enhancing the performance of the proposed framework.

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

## **Description of the scripts**

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
