import pandas as pd
import re
import sys

def read_data(filename):
    reviews = []
    with open(filename, 'r+') as fr:
        end_of_review = 0
        for line in fr:
            line = re.sub(r'[:][\d]', " ", str(line))
            if (re.search("#label#:negative", str(line))):
                line = re.sub("#label#:negative", " " ,str(line))
                end_of_review=1
            if (re.search("#label#:positive", str(line))):
                line = re.sub("#label#:positive", " " ,str(line))
                end_of_review=1    
            if end_of_review == 1:    
                reviews.append(str(line))
                end_of_review = 0           
        return reviews

def convert_to_dataframe(listname):
    return pd.DataFrame({'reviews':listname})

def get_label_from_filename(filename, df):
    if re.search("positive", filename):
        df["label"] = 1
    elif re.search("negative", filename):
        df["label"] = 0 
    return df

def add_word_count_column(df):
    df['#words'] = df.reviews.apply(lambda x: len(str(x).split(' ')))
    return df

def randomize_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)

def add_product_code(df, code):
    df["code"] = code
    return df

def load_dataset(path, product_code):
    neg_reviews_list = read_data(f'{path}/negative.review')
    df_neg = convert_to_dataframe(neg_reviews_list)
    df_neg = get_label_from_filename(f'{path}/negative.review', df_neg)

    pos_reviews_list = read_data(f'{path}/positive.review')
    df_pos = convert_to_dataframe(pos_reviews_list)
    df_pos = get_label_from_filename(f'{path}/positive.review', df_pos)

    df = pd.concat([df_neg, df_pos], axis=0)
    df = add_word_count_column(df)
    df = randomize_dataframe(df)
    df = add_product_code(df, product_code)
    
    return df

def main(base_path="."):
    # Load datasets    
    base_path = 'Downloads/processed_acl'
    df_books = load_dataset(f'{base_path}/books', "books")
    df_dvd = load_dataset(f'{base_path}/dvd', "dvd")
    df_kitchen = load_dataset(f'{base_path}/kitchen', "kitchen")
    df_electronics = load_dataset(f'{base_path}/electronics', "electronics")
    return df_books, df_dvd, df_kitchen, df_electronics

if __name__=='__main__':
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    main(base_path)


