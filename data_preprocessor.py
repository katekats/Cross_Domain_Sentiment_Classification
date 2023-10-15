import nltk

stop = set(('i','im','ive', 'me','my','myself','we','our','ours','ourselves','you','youre','youve','youll','youd','your','yours','yourself','yourselves','he','him','his','himself','she','shes','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','thatll','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','should','shouldve','now','d','ll','m','o','re','ve','y','ma'))

sno = nltk.stem.SnowballStemmer('english')

def replace_url(df,col,rm1,rm2):
    return(df[col].str.replace(rm1,rm2))

def extract_emo(df, col, emo):
    return(df[col].str.extractall(emo).unstack().apply(lambda x:' '.join(x.dropna()), axis=1))

def replace_emo(df,col,emo1,emo2):
    return(df[col].str.replace(emo1,emo2))

def replace_punct(df, col, punct1, punct2):
    return(df[col].str.replace(punct1, punct2))

def remove_numbers(df,col,rm1,rm2):
    return(df[col].str.replace(rm1,rm2))

def lower_words(df,col):
    return(df[col].apply(lambda x: " ".join(x.lower() for x in x.split())))

def remove_stop(df,col):
    return(df[col].apply(lambda x: " ".join(x for x in x.split() if x not in stop)))

def tokenize(df,col):
    return(df.apply(lambda row: nltk.word_tokenize(row[col]), axis=1))

def word_count(df,col):
    return(df[col].apply(lambda x: len(str(x).split(' '))))

def stemming(df,col):
    return(df[col].apply(lambda x: " ".join([sno.stem(word) for word in x.split()])))



def preprocess_data(df):
    df['nohtml'] = replace_url(df,'reviews','^http?:\/\/.*[\r\n]*','')
    df['nohtml'] = lower_words(df,'nohtml')
    df['nohtml'] = remove_numbers(df, 'nohtml', '[0-9]+',' ')
    df['nohtml'] = replace_punct(df, 'nohtml', '[^\w\s]',' ')
    df['nohtml'] = replace_punct(df, 'nohtml', '_',' ')
    df['nohtml'] = replace_punct(df, 'nohtml',r'\b(no|not|nt|dont|doesnt|doesn|don|didnt|cant|cannt|cannot|wouldnt|wont|couldnt|hasnt|havent|hadnt|shouldnt)\s+([a-z])',r'not \2')
    df['nohtml'] = remove_stop(df,'nohtml')
    df['tokenized'] = tokenize(df,'nohtml')
    df['#token'] = word_count(df,'tokenized')
    return df

