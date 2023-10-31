##Cross-Domain Sentiment Classification with Chi-square for Feaure Extraction and Ensembling

This project focuses on text classification using various feature extraction techniques, including TF-IDF, FastText embeddings, and Chi-squared statistics. Additionally, an ensemble learning approach using SuperLearner from mlxtend is employed.

## Getting Started
For running the framework, this work recommends creating a new virtual environment that uses Python version 3.7.7.
Afterward, you can install the packages in the requirements.txt of the requirements_files directory to get started. Using anaconda, the commands look like this:
```bash
conda create -n myenv python=3.7.7
conda activate myenv
conda install --file requirements_files/requirements.txt

**Steps for running the scripts**  

├── data_loader.py          # Script to load and process raw data
├── data_preprocessor.py    # Script containing data preprocessing functions
├── feature_extractor.py    # Script for feature extraction and embeddings
├── classifiers.py          # Script containing classifier pipelines and ensemble model
├── main.py                 # Main script to execute the project
└── README.md               # Documentation and usage instructions


For running the feature extractor, we need to give the name of the domains we want to use as source and target domains:
**python feature_extractor.py --dataset_key xy**

For example, we give **db** for using "DVD" as the source domain and "books" as the target domain: 
**python feature_extractor.py --dataset_key db**

The possible combinations are:

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
        'dk': ('dvd','kitchen')
        
    }

**Description of the scripts**
* **data_loader.py**: Contains the process_data function, which loads and processes the raw data for various product categories and sentiments.
* **data_preprocessor.py**: This module offers the preprocess_data function which prepares data for feature extraction.
* **feature_extractor.py**: The main feature extraction module. It contains functions like extract_important_features for Chi-squared feature extraction, exclude_nouns to filter out noun words, and df_to_fasttext_data for generating FastText embeddings.
* **classifiers.py**: Holds the classifier pipeline creation and ensemble model training logic. Includes various embedding vectorizers and a stacking ensemble model approach.
* **main.py**: The main executable script that brings everything together. Loads data, processes it, extracts features, and trains the ensemble model.


This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE.
