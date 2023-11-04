import nltk
import pandas as pd
from functools import partial
import data_loader

# Set of English stop words
stop = set(('i','im','ive', 'me','my','myself','we','our','ours','ourselves','you','youre','youve','youll','youd','your','yours','yourself','yourselves','he','him','his','himself','she','shes','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','thatll','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','should','shouldve','now','d','ll','m','o','re','ve','y','ma'))


# Initialize a stemmer object using NLTK's SnowballStemmer
sno = nltk.stem.SnowballStemmer('english')

# Function to remove URLs from text
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# Function to convert text to lowercase
def lower_text(text):
    return text.lower()

# Function to remove numeric characters from text
def remove_numbers(text):
    return re.sub('[0-9]+', '', text)

# Function to remove punctuation from text
def remove_punctuations(text):
    return re.sub('[^\w\s]', ' ', text)

# Function to remove stopwords from text
def remove_stopwords(text):
    return " ".join(word for word in text.split() if word.lower() not in stop)

# Function to stem words in text
def stem_text(text):
    return " ".join(sno.stem(word) for word in text.split())

# Function to tokenize text
def tokenize_text(text):
    return nltk.word_tokenize(text)

# Function to count words in text
def word_count(text):
    return len(text.split())

# Function to preprocess a dataframe column based on the above text operations
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

# Function to filter dataframe based on a minimum number of tokens
def filter_dataset_by_token_count(df, token_count):
    return df[df["#token"] > token_count].reset_index(drop=True)

# Function to preprocess each dataset and filter them based on token count criteria
def preprocess_datasets(df_books, df_dvd, df_kitchen, df_electronics):
    # Load datasets
    
    # Dictionary to hold dataframes corresponding to each category
    data_dict = {
        'books': df_books,
        'dvd': df_dvd,
        'electronics': df_electronics,
        'kitchen': df_kitchen
    }

# Criteria for the minimum number of tokens for each categ  ory
    criteria = {
        'books': 80,
        'dvd': 75,
        'electronics': 55,
        'kitchen': 54
    }

    # Process each dataframe and filter them according to the predefined token count criteria
    for key, df in data_dict.items():
        df = preprocess_dataframe(df, 'reviews')
        df = filter_dataset_by_token_count(df, criteria[key])
        data_dict[key] = df
        
    # Combinations dictionary to pair product categories for comparison or other operations
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


