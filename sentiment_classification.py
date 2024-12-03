import pandas as pd
import matplotlib.pyplot as plt


def classify_sentiment(score):
    """
    Classify sentiment based on the sentiment score.
    :param score: Sentiment score (float)
    :return: 'Positive', 'Neutral', or 'Negative'
    """
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_text_length_by_sentiment(data, text_column='Statement', sentiment_column='Sentiment Category'):
    """
    Analyze the relationship between text length and sentiment category.
    :param data: DataFrame containing text and sentiment columns
    :param text_column: Name of the column containing text
    :param sentiment_column: Name of the column containing sentiment categories
    :return: DataFrame of average text length by sentiment category
    """
    # Calculate text length
    data['Text Length'] = data[text_column].apply(len)
    avg_text_length = data.groupby(sentiment_column)['Text Length'].mean()
    print("Average Text Length by Sentiment Category:\n", avg_text_length)

    # Visualization
    ax = avg_text_length.plot(kind='bar', figsize=(8, 5), color='gray', zorder=3)
    plt.title('Average Text Length by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length')
    plt.xticks(rotation=0)
    ax.set_facecolor('#f7f7f7')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    plt.show()
    return avg_text_length
