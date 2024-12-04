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


def load_liar_dataset(file_path):
    """
    Load and preprocess the LIAR dataset.
    """
    liar_data = pd.read_csv(file_path, sep='\t', header=None)
    liar_data.columns = [
        'ID', 'Label', 'Statement', 'Subject', 'Speaker',
        'Job Title', 'State', 'Party', 'Pants on Fire',
        'False', 'Barely True', 'Half True', 'Mostly True',
        'Source'
    ]
    liar_data = liar_data[['Label', 'Statement']]
    liar_data = liar_data[liar_data['Label'].notnull()]
    return liar_data


def load_sentiment140_dataset(file_path):
    """
    Load and preprocess the Sentiment140 dataset.
    """
    sentiment_data = pd.read_csv(file_path, encoding='latin1', header=None)
    sentiment_data.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
    sentiment_data = sentiment_data[['Polarity', 'Text']]
    sentiment_data['Polarity'] = sentiment_data['Polarity'].map({0: 'Negative', 4: 'Positive'})
    return sentiment_data


def analyze_sentiment(data, column):
    """
    Perform sentiment analysis using Vader SentimentIntensityAnalyzer.
    """
    sia = SentimentIntensityAnalyzer()
    data['Sentiment'] = data[column].apply(
        lambda x: 'Positive' if sia.polarity_scores(x)['compound'] > 0 else 'Negative'
    )
    return data


def compute_word_count(data, column):
    """
    Compute word count for a given column.
    """
    data['Word Count'] = data[column].apply(lambda x: len(x.split()))
    return data


def train_naive_bayes_model(X_train, y_train):
    """
    Train a Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = model.predict(X_test)
    print("Accuracy for Misinformation Detection:", accuracy_score(y_test, y_pred))
    print("Classification Report for Misinformation Detection:\n", classification_report(y_test, y_pred))


def save_dataset(data, file_name):
    """
    Save the dataset to a CSV file.
    """
    data.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")


def calculate_sentiment_scores(data, column_name):
    """Calculate sentiment scores for a given text column."""
    sia = SentimentIntensityAnalyzer()
    data['Sentiment Score'] = data[column_name].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data


def visualize_sentiment_comparison(false_scores, true_scores):
    """Visualize sentiment score comparison between false and true statements."""
    false_avg_score = false_scores.mean()
    true_avg_score = true_scores.mean()

    # Prepare data for bar chart
    sentiment_comparison = pd.DataFrame({
        'Label': ['False Statements', 'True Statements'],
        'Average Sentiment Score': [false_avg_score, true_avg_score]
    })

    # Bar chart
    sentiment_comparison.plot(
        kind='bar', x='Label', y='Average Sentiment Score',
        color=['gray', 'black'], figsize=(8, 5), legend=False
    )
    plt.title('Average Sentiment Score for False vs True Statements')
    plt.xlabel('Statement Type')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=0)
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(false_scores.index, false_scores, color='black', label='False Statements', alpha=0.7)
    plt.scatter(true_scores.index, true_scores, color='gray', label='True Statements', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Neutral Sentiment')
    plt.title('Sentiment Score Distribution for False and True Statements')
    plt.xlabel('Index')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    plt.show()


def feature_engineering(data, text_column):
    """Add features for text length and word count."""
    data['Text Length'] = data[text_column].apply(len)
    data['Word Count'] = data[text_column].apply(lambda x: len(x.split()))
    return data


def train_random_forest(X, y):
    """Train a Random Forest model and evaluate its performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred), "\n")
    return rf_model


def visualize_feature_importance(rf_model, feature_columns):
    """Visualize the importance of features."""
    importances = pd.Series(rf_model.feature_importances_, index=feature_columns)
    importances.sort_values().plot(kind='barh', color='gray', figsize=(8, 5))
    plt.title('Feature Importance in Predicting False Statements')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    plt.show()

