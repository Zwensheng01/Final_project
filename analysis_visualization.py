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


def load_processed_dataset(file_path):
    """
    Load a processed dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def sentiment_distribution_by_label(data, label_col, sentiment_col):
    sentiment_summary = data.groupby(label_col)[sentiment_col].value_counts(normalize=True).unstack()
    print(f"Sentiment Distribution by {label_col}:\n", sentiment_summary)

    # Visualization
    ax = sentiment_summary.plot(kind='bar', stacked=True, figsize=(10, 6), color=['purple', 'grey'], zorder=3)
    plt.title(f'Sentiment Distribution by {label_col}')
    plt.xlabel(label_col)
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment', loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)  # 网格线在背景
    plt.show()
    return sentiment_summary


def average_word_count_by_sentiment(data, sentiment_col, word_count_col):
    avg_word_count = data.groupby(sentiment_col)[word_count_col].mean()
    print(f"Average Word Count by {sentiment_col}:\n", avg_word_count)

    # Visualization
    ax = avg_word_count.plot(kind='bar', figsize=(8, 5), color=['purple', 'gray'], zorder=3)
    plt.title(f'Average Word Count by {sentiment_col}')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Word Count')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)  # 网格线在背景
    plt.show()
    return avg_word_count


def average_word_count_by_label(data, label_col, word_count_col):
    avg_word_count = data.groupby(label_col)[word_count_col].mean()
    print(f"Average Word Count by {label_col}:\n", avg_word_count)

    # Visualization
    ax = avg_word_count.plot(kind='bar', figsize=(10, 6), color='grey', zorder=3)
    plt.title(f'Average Word Count by {label_col}')
    plt.xlabel(label_col)
    plt.ylabel('Average Word Count')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)  # 网格线在背景
    plt.show()
    return avg_word_count

