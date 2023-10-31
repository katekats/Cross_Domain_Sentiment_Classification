import nltk
import pandas as pd
from functools import partial

stop = set(('i','im','ive', 'me','my','myself','we','our','ours','ourselves','you','youre','youve','youll','youd','your','yours','yourself','yourselves','he','him','his','himself','she','shes','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','thatll','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','should','shouldve','now','d','ll','m','o','re','ve','y','ma'))

sno = nltk.stem.SnowballStemmer('english')

  # assuming stop words are not defined before

def process_column(df, col, func, *args, **kwargs):
    df[col] = df[col].apply(func, args=args, **kwargs)
    return df

replace_in_column = partial(process_column, func=lambda x, pattern, sub: x.replace(pattern, sub))
lower_column = partial(process_column, func=str.lower)
remove_from_column = partial(process_column, func=lambda x, pattern: x.replace(pattern, ''))
remove_stopwords = partial(process_column, func=lambda x: " ".join(word for word in x.split() if word not in stop))
stem_column = partial(process_column, func=lambda x: " ".join(sno.stem(word) for word in x.split()))
tokenize_column = partial(process_column, func=nltk.word_tokenize)
word_count_column = partial(process_column, func=lambda x: len(x.split()))

def preprocess_dataframe(df, col_name):
    df = replace_in_column(df, col_name, '^http?:\/\/.*[\r\n]*', '')
    df = lower_column(df, 'nohtml')
    df = remove_from_column(df, 'nohtml', '[0-9]+')
    
    punctuations = ['[^\w\s]', '_', r'\b(no|not|nt|dont|doesnt|doesn|don|didnt|cant|cannt|cannot|wouldnt|wont|couldnt|hasnt|havent|hadnt|shouldnt)\s+([a-z])']
    for punct in punctuations:
        df = replace_in_column(df, 'nohtml', punct, ' ')
    
    df = remove_stopwords(df, 'nohtml')
    df = tokenize_column(df, 'nohtml')
    df = word_count_column(df, 'tokenized')
    return df

def filter_dataset_by_token_count(df, token_count):
    return df[df["#token"] > token_count].reset_index(drop=True)

def main():
    global sample_df
    
    sample_df = preprocess_dataframe(sample_df, 'reviews')
    
    criteria = {
        'books': 80,
        'dvd': 75,
        'electronics': 55,
        'kitchen': 54
    }
    
    datasets = {key: filter_dataset_by_token_count(sample_df, value) for key, value in criteria.items()}
    
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
    
    combined_datasets = {key: datasets[df1].append(datasets[df2], ignore_index=True) for key, (df1, df2) in combinations.items()}
    
    sample_df = combined_datasets.get('db').copy()

if __name__ == '__main__':
    main()

