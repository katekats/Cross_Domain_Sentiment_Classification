import pandas as pd
import re
import sys

import re
import pandas as pd
import sys

# Function to read reviews from a file and return a list of reviews
def read_data(filename):
    reviews = []
    with open(filename, 'r+') as fr:
        end_of_review = 0
        for line in fr:
            # Replace colon followed by a digit with a space
            line = re.sub(r'[:][\d]', " ", str(line))
            # Check for the negative label and remove it
            if (re.search("#label#:negative", str(line))):
                line = re.sub("#label#:negative", " " ,str(line))
                end_of_review=1
            # Check for the positive label and remove it
            if (re.search("#label#:positive", str(line))):
                line = re.sub("#label#:positive", " " ,str(line))
                end_of_review=1    
            # If a label has been found, append the review to the list and reset the flag
            if end_of_review == 1:    
                reviews.append(str(line))
                end_of_review = 0           
        return reviews

# Convert a list of reviews into a DataFrame
def convert_to_dataframe(listname):
    return pd.DataFrame({'reviews': listname})

# Assign a label to the reviews based on the filename
def get_label_from_filename(filename, df):
    if re.search("positive", filename):
        df["label"] = 1  # Label for positive reviews
    elif re.search("negative", filename):
        df["label"] = 0  # Label for negative reviews
    return df

# Add a column to the DataFrame that contains the word count for each review
def add_word_count_column(df):
    df['#words'] = df.reviews.apply(lambda x: len(str(x).split(' ')))
    return df

# Shuffle the DataFrame rows and reset the index
def randomize_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)

# Add a 'code' column to the DataFrame to indicate the product category
def add_product_code(df, code):
    df["code"] = code
    return df

# Load the dataset from a given path and add necessary information
def load_dataset(path, product_code):
    # Read negative and positive reviews and convert them to DataFrames
    neg_reviews_list = read_data(f'{path}/negative.review')
    df_neg = convert_to_dataframe(neg_reviews_list)
    df_neg = get_label_from_filename(f'{path}/negative.review', df_neg)

    pos_reviews_list = read_data(f'{path}/positive.review')
    df_pos = convert_to_dataframe(pos_reviews_list)
    df_pos = get_label_from_filename(f'{path}/positive.review', df_pos)

    # Concatenate negative and positive reviews into one DataFrame
    df = pd.concat([df_neg, df_pos], axis=0)
    # Add word count column, randomize rows, and add product code
    df = add_word_count_column(df)
    df = randomize_dataframe(df)
    df = add_product_code(df, product_code)
    
    return df

# Main function to load all datasets
def main(base_path="."):
    # Set the base path for dataset directories and load each dataset
    base_path = 'processed_acl'
    df_books = load_dataset(f'{base_path}/books', "books")
    df_dvd = load_dataset(f'{base_path}/dvd', "dvd")
    df_kitchen = load_dataset(f'{base_path}/kitchen', "kitchen")
    df_electronics = load_dataset(f'{base_path}/electronics', "electronics")
    return df_books, df_dvd, df_kitchen, df_electronics

# Entry point of the script
if __name__=='__main__':
    # Get the base path from the command line argument if provided, otherwise use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    main(base_path)



