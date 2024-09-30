#!/usr/bin/env python3

import pandas as pd
import numpy as np
import nltk
import re
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'training.1600000.processed.noemoticon.csv')

if not os.path.exists(data_path):
    print(f"Dataset not found at {data_path}")
    print("Please ensure the dataset is placed in the 'data' directory relative to this script.")
    sys.exit(1)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the dataset
# Replace 'sentiment140.csv' with the path to your dataset
data = pd.read_csv(data_path, encoding='ISO-8859-1', header=None)

# Assign column names
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Keep only necessary columns
data = data[['text', 'sentiment']]

# Map sentiment values to 0 (negative) and 1 (positive)
data['sentiment'] = data['sentiment'].map({0: 0, 4: 1})

# Data preprocessing function
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    # Join tokens back to string
    return ' '.join(stemmed_tokens)

# Apply preprocessing to the text data
data['processed_text'] = data['text'].apply(preprocess_text)

# Feature extraction using TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
y = data['sentiment'].values

# For demonstration purposes, use a smaller subset
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Function to predict sentiment of new tweets
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return sentiment

# Example usage
if __name__ == "__main__":
    test_tweet = input("Enter a tweet to analyze its sentiment: ")
    result = predict_sentiment(test_tweet)
    print(f"The sentiment of the tweet is: {result}")
