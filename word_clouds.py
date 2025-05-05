import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ========================================
#              WORD CLOUDS
# ========================================
# just visualizations!


# read in all data
cbs = pd.read_csv('cbs_harvard.csv')
dw = pd.read_csv('dw_harvard.csv')
fox = pd.read_csv('fox_harvard.csv')
cnn = pd.read_csv('cnn_harvard.csv')

# put them in arrays
dfs = [cbs, dw, fox, cnn]
names = ["CBS", "DW", "Fox", "CNN"]

# get the stop words 
stop_words = set(stopwords.words('english')) 

# preprocess strings (tokenize, lemmatize, remove stop words)
def preprocess(sentence):
    if not isinstance(sentence, str):
        return ''

    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # split sentence into words
    word_tokens = word_tokenize(sentence) 

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
    
    # split sentence into words
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # return " ".join(filtered_sentence)
    return filtered_sentence

# get n-grams (helper)
def get_grams(text, n):
    if not isinstance(text, str):
        return []
    
    # Lowercase + remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    if n < 1:
        return []
    
    if n == 1:
        return [(word,) for word in words]  # single-word tuples
    
    ngrams = zip(*[words[i:] for i in range(n)])
    return list(ngrams)


# ========================================
#             CALL FUNCTIONS
# ========================================

# preprocess for all descriptions and put them all together
all_words = {}

for i in range(4):
    all_words = []
    for comment in dfs[i]['comment']:
        words = preprocess(comment)
        all_words.extend(words)

    all_text = " ".join(all_words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title(f"{names[i]} Word Cloud")
    plt.show()
