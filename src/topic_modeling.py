# topic_modeling.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf_matrix(df, min_df=0.1, max_df=1.0):
    """
    Calculate the TF-IDF matrix for a given set of text data.

    Parameters:
        text_data (list or Series): List or Series of text data.
        min_df (float, optional): Minimum document frequency for words to be included in the vocabulary.
        max_df (float, optional): Maximum document frequency for words to be included in the vocabulary.

    Returns:
        pd.DataFrame: TF-IDF matrix.
    """
    text_data = df['review_clean'].fillna("")
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    bow_representation = vectorizer.fit_transform(text_data)
    tfidf_df = pd.DataFrame(bow_representation.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df

def perform_kmeans_clustering(tfidf_df, num_clusters=5, random_state=42):
    """
    Perform K-means clustering on the TF-IDF matrix.

    Parameters:
        tfidf_df (pd.DataFrame): TF-IDF matrix.
        num_clusters (int, optional): Number of clusters for K-means.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with cluster labels added.
    """
    X = tfidf_df.copy()
    km = KMeans(n_clusters=num_clusters, n_init=200, random_state=random_state)
    km.fit(X)
    labels = km.predict(X)
    X['cluster'] = labels
    return X

def visualize_cluster_distribution(X):
    """
    Visualize the distribution of clusters.

    Parameters:
        X (pd.DataFrame): DataFrame with cluster labels.
    """
    cluster_counts = X['cluster'].value_counts()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    cluster_counts.plot(kind='bar', ax=ax)
    
    plt.xticks(rotation=0)
    plt.ylabel('count')
    plt.xlabel('cluster')
    plt.title('Cluster distribution')

    # Use st.pyplot() instead of plt.show() for Streamlit
    st.pyplot(fig)


def top_tfidf_means(X):
    """
    Display a bar chart of the top mean TF-IDF scores across all clusters.

    Parameters:
        X (pd.DataFrame): DataFrame with cluster labels.
    """
    counts = X.drop('cluster', axis=1).mean()
    st.bar_chart(counts)

def add_cluster_column(df, X):
    """
    Add a 'cluster' column to the original DataFrame.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        X (pd.DataFrame): DataFrame with cluster labels.
    """
    df['cluster'] = X['cluster']

def get_top_tfidf_means_by_cluster(X, cluster_number, top_n=10):
    """
    Get the top mean TF-IDF scores for a specific cluster.

    Parameters:
        X (pd.DataFrame): DataFrame with cluster labels.
        cluster_number (int): Cluster number for which to get top scores.
        top_n (int, optional): Number of top scores to retrieve.

    Returns:
        pd.Series: Top mean TF-IDF scores for the specified cluster.
    """
    n = cluster_number
    return X.query(f"cluster=={n}").drop('cluster', axis=1).mean().sort_values(ascending=False).head(top_n)

def topic_modeling(df):
    # Calculate TF-IDF matrix
    tfidf_df = calculate_tfidf_matrix(df)

    # Perform K-means clustering
    num_clusters = 5
    X = perform_kmeans_clustering(tfidf_df, num_clusters)

    # Add new cluster column to the original data
    add_cluster_column(df, X)

    return df
