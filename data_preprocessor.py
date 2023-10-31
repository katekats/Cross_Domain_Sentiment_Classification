import nltk
import pandas as pd
from functools import partial
import data_loader

stop = set(('i','im','ive', 'me','my','myself','we','our','ours','ourselves','you','youre','youve','youll','youd','your','yours','yourself','yourselves','he','him','his','himself','she','shes','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','thatll','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','should','shouldve','now','d','ll','m','o','re','ve','y','ma'))

sno = nltk.stem.SnowballStemmer('english')

def remove_url(text):
    return re.sub('^http?:\/\/.*[\r\n]*', '', text)

def lower_text(text):
    return text.lower()

def remove_numbers(text):
    return re.sub('[0-9]+', '', text)

def remove_punctuations(text):
    return re.sub('[^\w\s]', ' ', text)

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stop)

def stem_text(text):
    return " ".join(sno.stem(word) for word in text.split())

def tokenize_text(text):
    return nltk.word_tokenize(text)

def word_count(text):
    return len(text.split())

def preprocess_dataframe(df, col_name):
    df[col_name] = (df[col_name].apply(remove_url)
                                    .apply(lower_text)
                                    .apply(remove_numbers)
                                    .apply(remove_punctuations)
                                    .apply(remove_stopwords)
                                    .apply(stem_text))
    df["#token"] = df[col_name].apply(tokenize_text)
    df["word_count"] = df[col_name].apply(word_count)
    return df

def filter_dataset_by_token_count(df, token_count):
    return df[df["#token"] > token_count].reset_index(drop=True)

def preprocess_datasets(df_books, df_dvd, df_kitchen, df_electronics):
    # Load datasets
    

    data_dict = {
        'books': df_books,
        'dvd': df_dvd,
        'electronics': df_electronics,
        'kitchen': df_kitchen
    }

    criteria = {
        'books': 80,
        'dvd': 75,
        'electronics': 55,
        'kitchen': 54
    }

    # Preprocess each dataset and filter based on criteria
    for key, df in data_dict.items():
        df = preprocess_dataframe(df, 'reviews')
        df = filter_dataset_by_token_count(df, criteria[key])
        data_dict[key] = df

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

    return data_dict, combinations

if __name__ == "__main__":
    # Load the data
    df_books, df_dvd, df_kitchen, df_electronics = data_loader.main()

    # Preprocess the datasets
    datasets, combinations = preprocess_datasets(df_books, df_dvd, df_kitchen, df_electronics)

    # Print or check results for testing
    print(datasets["books"])


