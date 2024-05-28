from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk import word_tokenize, bigrams
nltk.download('punkt')

def unigram_frequency_cluster_top(df, cluster, top):
    df_subset = df[df['cluster'] == cluster]
    # Convert 'review_clean' column to strings
    df_subset['review_clean'] = df_subset['review_clean'].astype(str)
    # Join text together
    text_values = ','.join(list(df_subset['review_clean']))
    # Count each word
    Counter_values = Counter(text_values.split())
    most_frequent_values = Counter_values.most_common(top)
    most_frequent_values = pd.DataFrame(most_frequent_values, columns=['Word', 'Count'])
    return most_frequent_values

def bigram_frequency_cluster_top(df, cluster, top):
    df_subset = df[df['cluster'] == cluster]
    # Assuming 'reviews' is your DataFrame and 'review_clean' is the column with text data
    df_subset_list = list(df_subset['review_clean'].fillna("").values)
    # Initialize an empty list to store all bigrams
    all_bigrams = []
    # Iterate over each review in the list
    for review in df_subset_list:
        # Tokenize the review into words
        words = nltk.tokenize.wordpunct_tokenize(review)
        # Generate bigrams from the list of words
        bi_grams = list(bigrams(words))
        # Convert bigrams to strings and join the words with a space
        bi_grams_str = [' '.join(gram) for gram in bi_grams]
        # Extend the list of all bigrams with the bigrams from the current review
        all_bigrams.extend(bi_grams_str)
    # Count occurrences of each bigram
    counter_bigrams = Counter(all_bigrams)
    most_frequent_bigrams = counter_bigrams.most_common(top)
    # Create DataFrame for most frequent bigrams
    most_frequent_bigrams = pd.DataFrame(most_frequent_bigrams, columns=['Bigram', 'Count'])
    return most_frequent_bigrams