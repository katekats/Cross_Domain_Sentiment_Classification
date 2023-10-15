from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from nltk.tag import pos_tag
from fastText import load_model


def extract_important_features(df, text_column, target_column, n_features=1000):
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[text_column].values)
    feature_names = count_vect.get_feature_namessample_df = sample_df1.copy()()   
    chi2_scores = chi2(X, df[target_column])
    sorted_scores = sorted(zip(feature_names, chi2_scores[0]), key=lambda x:x[1])[-n_features:]
    return [k for k, v in sorted_scores if v < 0.05]

def exclude_nouns(token_list):
    return [word for word, pos in pos_tag(token_list) if not pos.startswith('N')]


def df_to_fasttext_data(df, token_column, model):
    all_words = set(w for tokens in df[token_column] for w in tokens)
    return {word: model.get_word_vector(word).astype('float32') for word in all_words}

# Load fastText model (uncomment the appropriate line)
# ft_model = FastText("Downloads/cc.en.300.bin")
# ft_model = fastText.load_model('Downloads/cc.en.300.bin')
ft_model = load_model('Downloads/cc.en.300.bin')

# Extract important features based on Chi-squared scores
cat_features = extract_important_features(sample_df, 'nohtml', 'code')
label_features = extract_important_features(sample_df, 'nohtml', 'label')

# Exclude nouns and create a new column 'tokenized2'
sample_df["tokenized2"] = sample_df["tokenized"].apply(exclude_nouns)

# Extract FastText vectors
fasttext_dict = df_to_fasttext_data(sample_df, 'tokenized', ft_model)
fasttext2_dict = df_to_fasttext_data(sample_df, 'tokenized2', ft_model)

