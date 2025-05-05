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

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

cbs_data = pd.read_csv('cbs_harvard.csv')
# print(cbs_data)

# exploratory graphs
# # number of likes
# plt.figure(figsize=(10,6))
# sns.histplot(cbs_data['likes'], bins=30)
# plt.title('Distribution of Comment Likes')
# plt.xlabel('Number of Likes')
# plt.ylabel('Number of Comments')
# plt.show()

# # number of comments over time
# cbs_data['published_at'] = pd.to_datetime(cbs_data['published_at'])
# cbs_data.set_index('published_at').resample('D').size().plot()
# plt.title('Number of Comments Over Time')
# plt.xlabel('Date')
# plt.ylabel('Number of Comments')
# plt.show()

print(cbs_data.columns)
filtered_popular = cbs_data[cbs_data['likes'] > 10]
filtered_popular = filtered_popular.reset_index(drop=True)
print(f"likes over 10: {len(filtered_popular)}")
# print(filtered_popular)

# get the stop words 
stop_words = set(stopwords.words('english')) 

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
    # print(filtered_sentence)
    # print(" ".join(filtered_sentence))
    return " ".join(filtered_sentence)

# def extract_emojis(text):
#     if not isinstance(text, str):
#         return []
#     # Use regex to grab emoji characters
#     emoji_list = [char for char in text if char in emoji.EMOJI_DATA]
#     return emoji_list

# # Extract list of emojis per comment
# filtered_popular['emojis'] = filtered_popular['comment'].apply(extract_emojis)

# from collections import Counter

# # Flatten all emojis into a single list
# all_emojis = [emoji for sublist in filtered_popular['emojis'] for emoji in sublist]

# # Count frequency
# emoji_counts = Counter(all_emojis)

# # Print top emojis
# print(emoji_counts.most_common(10))

cbs_data['cleaned_comment'] = cbs_data['comment'].apply(preprocess)
# print(cbs_data['cleaned_comment'])

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

# cbs_data['monograms'] = cbs_data['comment'].apply(lambda x: get_grams(x, 1))

# cbs_monos = []
# cbs_bis = []
# for comment in cbs_data['comment']:
#     comment = preprocess(comment)
#     for gram in get_grams(comment, 2):
#         cbs_bis.append(' '.join(gram) )

# for comment in cbs_data['comment']:
#     comment = preprocess(comment)
#     for gram in get_grams(comment, 1):
#         cbs_monos.append(' '.join(gram) )

# ngrams_bis = Counter(cbs_bis)
# ngrams_monos = Counter(cbs_monos)
# top_ngrams_bis = ngrams_bis.most_common(40)  # top 10 n-grams
# top_ngrams_monos = ngrams_monos.most_common(40)  # top 10 n-grams
# ngrams_bis, counts_bis = zip(*top_ngrams_bis)
# ngrams_monos, counts_monos = zip(*top_ngrams_monos)


# plt.figure(figsize=(10,6))
# plt.barh(ngrams_bis, counts_bis, color='skyblue')
# plt.xlabel('Count')
# plt.ylabel('N-gram')
# plt.title('Top 20 Bigrams in YouTube Comments under CBS News')
# plt.gca().invert_yaxis()  # puts the biggest at the top
# plt.show()

# plt.figure(figsize=(10,6))
# plt.barh(ngrams_monos, counts_monos, color='skyblue')
# plt.xlabel('Count')
# plt.ylabel('N-gram')
# plt.title('Top 50 Monograms in YouTube Comments')
# plt.gca().invert_yaxis()  # puts the biggest at the top
# plt.show()

def process_and_return_graph(data, news, type):
    mono = []
    bi = []
    for comment in data['comment']:
        comment = preprocess(comment)
        for gram in get_grams(comment, 2):
            bi.append(' '.join(gram))
        for gram in get_grams(comment, 1):
            mono.append(' '.join(gram))
    ngrams_bis = Counter(bi)
    ngrams_monos = Counter(mono) 
    top_ngrams_bis = ngrams_bis.most_common(40)
    top_ngrams_monos = ngrams_monos.most_common(40)
    ngrams_bis, counts_bis = zip(*top_ngrams_bis)
    ngrams_monos, counts_monos = zip(*top_ngrams_monos)

    fig, axs = plt.subplots(figsize=(6, 6))
    if type == 1:
        plt.barh(ngrams_bis, counts_bis, color='skyblue')
        plt.title(f'Top 20 Bigrams in YouTube Comments under {news} News')
        plt.gca().invert_yaxis()  # puts the biggest at the top
    if type == 2:
        plt.barh(ngrams_monos, counts_monos, color='skyblue')
        plt.title(f'Top 50 Monograms in YouTube Comments under {news} News')
        plt.gca().invert_yaxis()  # puts the biggest at the top
    return fig, axs

# return all three together
def compare(graph1, graph2, graph3):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    axs[0] = graph1
    axs[1] = graph2
    axs[2] = graph3
    plt.tight_layout()
    plt.show()

cbs = pd.read_csv('cbs_harvard.csv')
dw = pd.read_csv('dw_harvard.csv')
fox = pd.read_csv('fox_harvard.csv')
# g1 = process_and_return_graph(cbs, "CBS", 2)
# g2 = process_and_return_graph(dw, "DW", 2)
# g3 = process_and_return_graph(fox, "FOX", 2)
# compare(g1, g2, g3)

def compare_three(d1, d2, d3, names):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, data, name in zip(axs, [d1, d2, d3], names):
        mono = []
        for comment in data['comment']:
            comment = preprocess(comment)
            for gram in get_grams(comment, 2):
                mono.append(' '.join(gram))
        ngram_counts = Counter(mono).most_common(40)
        ngrams, counts = zip(*ngram_counts)
        
        ax.barh(ngrams, counts, color='skyblue')
        ax.set_title(f"Top 40 Monograms under {name}")
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

compare_three(cbs, dw, fox, ["CBS", "DW", "FOX"])