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

def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'
        
def analyze_sentiment_by_topic(data, topic_column='Topic', sentiment_column='Sentiment Score'):
    """
    Analyze sentiment distribution by topic.
    """
    # Classify sentiment
    data['Sentiment Category'] = data[sentiment_column].apply(classify_sentiment)

    # Calculate sentiment distribution by topic
    sentiment_by_topic = data.groupby(topic_column)['Sentiment Category'].value_counts(normalize=True).unstack()
    print("Sentiment Distribution by Topic:\n", sentiment_by_topic)

    # Visualization
    ax = sentiment_by_topic.plot(kind='bar', stacked=True, figsize=(10, 6), color=['purple', 'grey', 'black'], zorder=3)
    plt.title('Sentiment Distribution by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment')

    # Background and gridline adjustments
    ax.set_facecolor('#f7f7f7')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    plt.show()


def analyze_false_statement_proportion(data, label_column='Label', topic_column='Topic'):
    """
    Analyze and visualize the proportion of false statements by topic.
    """
    false_statements = data[data[label_column].isin(['false', 'pants-fire'])]
    false_statements_by_topic = false_statements.groupby(topic_column).size()
    total_statements_by_topic = data.groupby(topic_column).size()
    false_ratio_by_topic = (false_statements_by_topic / total_statements_by_topic).fillna(0)

    print("False Statement Ratio by Topic:\n", false_ratio_by_topic)

    # Visualization
    ax = false_ratio_by_topic.plot(kind='bar', figsize=(8, 5), color='gray', zorder=3)
    plt.title('Proportion of False Statements by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Proportion of False Statements')

    # Background and gridline adjustments
    ax.set_facecolor('#f7f7f7')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    plt.show()


def analyze_sentiment_by_label_and_topic(data, topic_column='Topic', label_column='Label', sentiment_column='Sentiment Score'):
    """
    Analyze average sentiment scores by label and topic.
    """
    avg_sentiment_by_label_and_topic = data.groupby([topic_column, label_column])[sentiment_column].mean().unstack()
    print("Average Sentiment by Label and Topic:\n", avg_sentiment_by_label_and_topic)

    # Visualization
    ax = avg_sentiment_by_label_and_topic.plot(kind='bar', figsize=(12, 6), color=['gray', 'black'], zorder=3)
    plt.title('Average Sentiment by Label and Topic')
    plt.xlabel('Topic')
    plt.ylabel('Average Sentiment Score')
    plt.legend(title='Label')

    # Background, gridline, and baseline adjustments
    ax.set_facecolor('#f7f7f7')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=1)  # Baseline
    plt.show()


def compare_topic_distribution(data, label_column='Label', topic_column='Topic'):
    """
    Compare the topic distribution between false and true statements.
    """
    false_statements = data[data[label_column].isin(['false', 'pants-fire'])]
    true_statements = data[data[label_column] == 'true']

    false_topic_distribution = false_statements[topic_column].value_counts(normalize=True)
    true_topic_distribution = true_statements[topic_column].value_counts(normalize=True)

    # Visualization
    comparison = pd.DataFrame({'False Statements': false_topic_distribution, 'True Statements': true_topic_distribution})
    print("Topic Distribution Comparison:\n", comparison)
    ax = comparison.plot(kind='bar', figsize=(8, 5), color=['black', 'gray'], zorder=3)
    plt.title('Topic Distribution for False vs True Statements')
    plt.xlabel('Topic')
    plt.ylabel('Proportion')

    # Background and gridline adjustments
    ax.set_facecolor('#f7f7f7')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    plt.xticks(rotation=0)
    plt.show()