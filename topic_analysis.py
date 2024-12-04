import importlib
import matplotlib.pyplot as plt
import pandas as pd
import analysis_module as am
import analysis_visualization as av
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier


def perform_topic_modeling(data, text_column, ngram_range=(2, 3), n_topics=5, max_features=1000):
    """
    Perform topic modeling on text data using Latent Dirichlet Allocation (LDA).
    
    Args:
    - data (pd.DataFrame): The dataset containing the text data.
    - text_column (str): The column name of the text data to analyze.
    - ngram_range (tuple): The range of n-grams to consider for feature extraction (default is bi-grams and tri-grams).
    - n_topics (int): The number of topics to extract (default is 5).
    - max_features (int): The maximum number of features to include in the vectorization (default is 1000).
    
    Returns:
    - pd.DataFrame: Updated DataFrame with an additional 'Topic' column.
    - list: A list of top keywords for each topic.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features)
    X_ngrams = vectorizer.fit_transform(data[text_column])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_ngrams)

    # Extract topic keywords
    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, topic in enumerate(lda.components_):
        keywords = [terms[j] for j in topic.argsort()[-10:]]
        topics.append(keywords)
        print(f"Topic {i+1}: {' '.join(keywords)}")

    # Assign topics to each text
    data['Topic'] = lda.transform(vectorizer.transform(data[text_column])).argmax(axis=1)
    return data, topics
