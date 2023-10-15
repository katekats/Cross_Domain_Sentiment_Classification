import pandas as pd
import re

STOPWORDS = set(('i', 'im', 'ive', ...))
SNO = nltk.stem.SnowballStemmer('english')

def read_data(filename):
    reviews = []
    with open(filename, 'r') as fr:
        end_of_review = False
        for line in fr:
            line = re.sub(r'[:][\d]', " ", line)
            if "#label#:negative" in line:
                line = line.replace("#label#:negative", "")
                end_of_review = True
            elif "#label#:positive" in line:
                line = line.replace("#label#:positive", "")
                end_of_review = True

            if end_of_review:
                reviews.append(line.strip())
                end_of_review = False
    return reviews

def process_data(path, sentiment):
    reviews_list = read_data(path + sentiment + '.review')
    df = pd.DataFrame({'reviews': reviews_list})
    df['label'] = 1 if sentiment == "positive" else 0
    df['#words'] = df['reviews'].apply(lambda x: len(x.split()))
    df['code'] = path.split("/")[-2]
    return df.sample(frac=1).reset_index(drop=True)
    
