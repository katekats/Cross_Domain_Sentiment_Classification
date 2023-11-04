import scipy as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import chi2_contingency, chi2
import data_loader
import nltk
import pandas as pd
import re
# Make sure to download the necessary NLTK resources:
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Functions for preprocessing steps
stop_words = set(stopwords.words('english'))  # You can add more stopwords to this set if required

def add_word_count_column(df, text_column_name):
    """
    Adds a column to the DataFrame that contains the word count of the reviews.
    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.
    text_column_name (str): The name of the column that contains the text to count words from.
    
    Returns:
    pandas.DataFrame: The DataFrame with the added word count column.
    """
    # Adding a new column that contains the word count for the reviews
    df['#words'] = df[text_column_name].apply(lambda x: len(str(x).split(' ')))
    return df

def shuffle_and_add_code(df, code_value):
    """
    Shuffles the DataFrame rows, resets the index, and adds a 'code' column.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to shuffle.
    code_value (str): The value to assign in the new 'code' column.
    
    Returns:
    pandas.DataFrame: The shuffled DataFrame with the new 'code' column.
    """
    # Shuffling the DataFrame and resetting the index
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    # Adding the 'code' column
    df_shuffled['code'] = code_value
    return df_shuffled


def preprocess_text(text):
    """
    Preprocess a single text string.
    
    Parameters:
    - text: The text to preprocess.
    
    Returns:
    - A preprocessed text string.
    """
    # Replace URLs with empty string
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuations with space
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    
    # Replace negations with 'not'
    negations_regex = re.compile(r'\b(no|not|nt|dont|doesnt|didnt|cant|cannt|cannot|wouldnt|wont|couldnt|hasnt|havent|hadnt|shouldnt)\s+([a-z])', re.IGNORECASE)
    text = negations_regex.sub(r'not \2', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join the words back into one string separated by space
    return ' '.join(tokens), len(tokens)

def apply_preprocessing(df, column_name):
    """
    Apply text preprocessing to a specific column in DataFrame.
    
    Parameters:
    - df: The DataFrame to process.
    - column_name: The name of the column to preprocess.
    
    Returns:
    - A new DataFrame with the processed column, token count, and tokenized text.
    """
    # Apply the preprocess_text function and get preprocessed text and tokens
    preprocessed_and_tokens = df[column_name].apply(preprocess_text)
    
    # Create columns for preprocessed text and token count
    df['preprocessed'] = preprocessed_and_tokens.apply(lambda x: x[0])
    df['token_count'] = preprocessed_and_tokens.apply(lambda x: x[1])
    
    # Create a 'tokenized' column which will store the actual tokens
    df['tokenized'] = preprocessed_and_tokens.apply(lambda x: nltk.word_tokenize(x[0]))
    
    return df





def pos_counts(df, token_column):
    # Tagging POS only once
    df['pos_tags'] = df[token_column].apply(pos_tag)
    
    # Create new columns for noun, verb, adjective, and adverb counts
    df['noun_count'] = df['pos_tags'].apply(lambda tags: sum(pos.startswith('N') for word, pos in tags))
    df['verb_count'] = df['pos_tags'].apply(lambda tags: sum(pos.startswith('VB') for word, pos in tags))
    df['adjective_count'] = df['pos_tags'].apply(lambda tags: sum(pos.startswith('JJ') for word, pos in tags))
    df['adverb_count'] = df['pos_tags'].apply(lambda tags: sum(pos.startswith('RB') for word, pos in tags))
    df['other_count'] = df['pos_tags'].apply(lambda tags: sum(not pos.startswith(('N', 'VB', 'JJ', 'RB')) for word, pos in tags))

    return df



def perform_chi_squared_test(table, category_name):
    stat, p, dof, expected = chi2_contingency(table)
    print(f'Chi-squared test for {category_name}')
    print(f'dof={dof}')
    print(f'Expected frequencies:\n{expected}')

    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print(f'probability={prob}, critical={critical}, stat={stat}')
    
    if abs(stat) >= critical:
        print('Dependent (reject H0) - Significant difference in {0} counts by category.'.format(category_name))
    else:
        print('Independent (fail to reject H0) - No significant difference in {0} counts by category.'.format(category_name))
    
    alpha = 1.0 - prob
    print(f'significance={alpha}, p={p}')
    
    if p <= alpha:
        print('Dependent (reject H0) - Significant difference in {0} counts by category.'.format(category_name))
    else:
        print('Independent (fail to reject H0) - No significant difference in {0} counts by category.'.format(category_name))
    print('\n')


def main():
    try:
        # Replace 'data_loader.main()' with the actual code to load your data
        df_books, df_dvd, df_kitchen, df_electronics = data_loader.main()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        df_books = add_word_count_column(df_books, 'reviews')
        df_books = shuffle_and_add_code(df_books, 'books')

        df_dvd = add_word_count_column(df_dvd, 'reviews')
        df_dvd = shuffle_and_add_code(df_dvd, 'dvd')

        df_kitchen = add_word_count_column(df_kitchen, 'reviews')
        df_kitchen = shuffle_and_add_code(df_kitchen, 'kitchen')

        df_electronics = add_word_count_column(df_electronics, 'reviews')
        df_electronics = shuffle_and_add_code(df_electronics, 'electronics')
    except KeyError as e:
        print(f"DataFrame does not contain the expected columns: {e}")
        return
    except Exception as e:
        print(f"Error processing dataframes: {e}")
        return

    # Create a combined sample DataFrame for demonstration; replace with actual data processing as needed
    try:
        # Before concatenation, add a 'category' column to each DataFrame
        df_books['category'] = 'Books'
        df_dvd['category'] = 'DVD'
        df_kitchen['category'] = 'Kitchen'
        df_electronics['category'] = 'Electronics'

# Concatenate the DataFrames and keep the 'category' column
        sample_df = pd.concat([df_books, df_dvd, df_kitchen, df_electronics], ignore_index=True)

        sample_df = pd.concat([df_books, df_dvd, df_kitchen, df_electronics], ignore_index=True)
        sample_df = apply_preprocessing(sample_df, 'reviews')
    # Now that we have the 'tokenized' column, we can call pos_counts
        sample_df = pos_counts(sample_df, 'tokenized')
    except Exception as e:
        print(f"Error applying preprocessing: {e}")
        return

    try:
        # Assuming you have a 'category' column that has the product category as 'books', 'dvd', etc.
        # Create contingency tables for noun and verb counts
        noun_count_table = pd.crosstab(sample_df['category'], sample_df['noun_count'])
        verb_count_table = pd.crosstab(sample_df['category'], sample_df['verb_count'])
        adjective_count_table = pd.crosstab(sample_df['category'], sample_df['adjective_count'])
        adverb_count_table = pd.crosstab(sample_df['category'], sample_df['adverb_count'])
        other_count_table = pd.crosstab(sample_df['category'], sample_df['other_count'])
        # Perform Chi-squared tests
        perform_chi_squared_test(noun_count_table, 'noun')
        perform_chi_squared_test(verb_count_table, 'verb')
        perform_chi_squared_test(noun_count_table, 'noun')
        perform_chi_squared_test(adjective_count_table, 'adjective')
        perform_chi_squared_test(adverb_count_table, 'adverb')
        perform_chi_squared_test(other_count_table, 'other')
    except ValueError as e:
        print(f"Error with Chi-squared test: {e}")
    except Exception as e:
        print(f"General error with statistical test: {e}")

if __name__ == "__main__":
    main()