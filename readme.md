Text Classification with Feature Engineering and Ensembling

This project focuses on text classification using various feature extraction techniques including TF-IDF, FastText embeddings, and Chi-squared statistics. Additionally, an ensemble learning approach using SuperLearner from mlxtend is employed.

Table of Contents

Requirements
File Structure
Usage
Modules Description
Contributing
License
Requirements

Python 3.x
scikit-learn
mlxtend
fastText
nltk
numpy
File Structure

bash
Copy code
.
├── data_loader.py          # Script to load and process raw data
├── data_preprocessor.py    # Script containing data preprocessing functions
├── feature_extractor.py    # Script for feature extraction and embeddings
├── classifiers.py          # Script containing classifier pipelines and ensemble model
├── main.py                 # Main script to execute the project
└── README.md               # This file
Usage

Clone the repository:
bash
Copy code
git clone [repository-url]
cd [repository-directory]
Ensure you have the required Python packages installed:
bash
Copy code
pip install -r requirements.txt
Execute the main script:
bash
Copy code
python main.py
Modules Description

data_loader.py: Contains the process_data function which loads and processes the raw data for various product categories and sentiments.
data_preprocessor.py: This module offers the preprocess_data function which prepares data for feature extraction.
feature_extractor.py: The main feature extraction module. Contains functions like extract_important_features for Chi-squared feature extraction, exclude_nouns to filter out noun words, and df_to_fasttext_data for generating FastText embeddings.
classifiers.py: Holds the classifier pipeline creation and ensemble model training logic. Includes various embedding vectorizers and a stacking ensemble model approach.
main.py: The main executable script that brings everything together. Loads data, processes it, extracts features, and trains the ensemble model.
Contributing

Fork the repository on GitHub.
Clone your fork, create a new branch, make your changes and improvements, and push them.
Create a pull request detailing your changes.
License

This project is licensed under the MIT License.
