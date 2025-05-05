import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
import re
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ========================================
#              EXPERIMENT TWO
# ========================================
# 1. Compares top n-grams from all news outlet
# 2. Preprocessing before extracting grams


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
    return " ".join(filtered_sentence)

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

# compare results from all 
def compare_all_top_grams(dfs, names, top_x, n):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    for ax, data, name in zip(axs, dfs, names):
        grams = []
        for comment in data['comment']:
            comment = preprocess(comment)
            for gram in get_grams(comment, n):
                grams.append(' '.join(gram))
        ngram_counts = Counter(grams).most_common(top_x)
        ngrams, counts = zip(*ngram_counts)
        
        ax.barh(ngrams, counts, color='skyblue')
        ax.set_title(f"Top {top_x} {n}-grams under {name}")
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# compare results from all, returns raw grams
def compare_all_top_grams(dfs, names, top_x, n):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    grams = {}
    
    for ax, data, name in zip(axs, dfs, names):
        grams[name] = []
        for comment in data['comment']:
            comment = preprocess(comment)
            for gram in get_grams(comment, n):
                grams[name].append(' '.join(gram))
        ngram_counts = Counter(grams[name]).most_common(top_x)
        ngrams, counts = zip(*ngram_counts)
        
        ax.barh(ngrams, counts, color='skyblue')
        ax.set_title(f"Top {top_x} {n}-grams under {name}")
        ax.invert_yaxis()

        grams[name] = Counter(grams[name])
    
    plt.tight_layout()
    plt.show()
    return grams

def build_gram_df(grams, top_x):

    all_grams = Counter()

    for counter in grams.values():
        all_grams.update(counter)

    top_grams = []
    for gram, count in all_grams.most_common(top_x):
        top_grams.append(gram)

    data = {}
    for gram in top_grams:
        for name, counter in grams.items():
            data[name] = [counter.get(gram, 0) for gram in top_grams]
    df = pd.DataFrame(data, index=top_grams)
    return df

def unique_grams(grams, names):
    unique = {}
    for name in names:
        for g, count in grams[name].items():
            if g not in unique:
                unique[g] = count
            else:
                unique[g]+= count

    # Get top 20 overall
    top_unique = Counter(unique).most_common(20)
    keys, counts = zip(*top_unique)
    
    plt.figure(figsize=(10,6))
    plt.barh(keys, counts, color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('N-gram')
    plt.title('Top 20 grams in all news outlets')
    plt.gca().invert_yaxis()  # puts the biggest at the top
    plt.show()


# ========================================
#             CALL FUNCTIONS
# ========================================

grams = compare_all_top_grams(dfs, names, 30, 2)
# unique_grams(grams, names)
df = build_gram_df(grams, 30)
print(df)

def plot_stacked_bar(df):
    df.plot(kind='barh', stacked=True, figsize=(10, 8), colormap='tab20')
    plt.xlabel('Count')
    plt.ylabel('N-gram')
    plt.title('N-gram Counts by News Outlet')
    plt.gca().invert_yaxis()  # put biggest at top
    plt.tight_layout()
    plt.show()

plot_stacked_bar(df)
