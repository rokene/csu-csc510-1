#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import string
import nltk
import sys
import joblib
import os
import logging
from logging.handlers import RotatingFileHandler

import config

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Deep Learning libraries
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

logger = None

# CONSTANTS

logo = """
███████╗███████╗███╗   ██╗████████╗██╗███╗   ███╗███████╗███╗   ██╗████████╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║   
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   
███████║███████╗██║ ╚████║   ██║   ██║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
                                                                            
 █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗             
██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗            
███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝            
██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗            
██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║            
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝            
"""

# FUNCTIONS

def setup_logging(log_file='app.log'):
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(config.log_level)  # Set the desired log level

    # Define log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create handlers
    # RotatingFileHandler rotates log files based on size
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.log_size_mb * 1024 * 1024,
        backupCount=config.log_num_rotated_files)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(config.log_level)

    # StreamHandler for printing to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(config.log_level)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords and lemmatize
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back to string
    text = ' '.join(tokens)
    return text


def load_data():
    # Load the Sentiment140 Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'training.1600000.processed.noemoticon.csv')

    if not os.path.exists(data_path):
        logger.info(f"Dataset not found at {data_path}")
        logger.info("Please ensure the dataset is placed in the 'data' directory relative to this script.")
        sys.exit(1)

    # Load the dataset
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(data_path, encoding='ISO-8859-1', names=columns)

    # Map the target values to 0 and 1
    df['target'] = df['target'].replace(4, 1)

    return df


# Data Preprocessing
def preprocess_data(df:pd.DataFrame):
    # Apply preprocessing to the text column
    logger.info("\nPreprocessing text data...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Display the first few cleaned texts
    logger.info("Cleaned text sample:")
    logger.info(df[['text', 'clean_text']].head())


# TF-IDF Vectorization for Traditional ML Models
def ml_model_idf_vectorization(df:pd.DataFrame, max_features, test_size, random_state):
    # Split data for traditional ML models
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
        df['clean_text'], df['target'], test_size=test_size, random_state=random_state)

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectorizer.fit(X_train_tfidf)

    # Transform the text data
    X_train_tfidf = tfidf_vectorizer.transform(X_train_tfidf)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_tfidf)

    return X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, tfidf_vectorizer


# Tokenization and Padding for Neural Network Models
def nn_model_tokenization_padding(df:pd.DataFrame, max_num_words, max_sequence_length, test_size, random_state):
    # Initialize Tokenizer
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(df['clean_text'])

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['clean_text'])

    # Pad sequences
    word_index = tokenizer.word_index
    data_seq = pad_sequences(sequences, maxlen=max_sequence_length)

    # Split data for neural network models
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        data_seq, df['target'], test_size=test_size, random_state=random_state)

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, tokenizer


# Training Machine Learning Model (Logistic Regression)
def ml_model_training_evaluation(X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, max_iter):
    logger.info("\nTraining Logistic Regression model...")

    # Initialize and train the model
    lr_model = LogisticRegression(max_iter=max_iter)
    lr_model.fit(X_train_tfidf, y_train_tfidf)

    # Make predictions
    y_pred_tfidf = lr_model.predict(X_test_tfidf)

    # Evaluate the model
    logger.info("\nLogistic Regression Performance:")
    logger.info(classification_report(y_test_tfidf, y_pred_tfidf))
    logger.info(f"Accuracy: {accuracy_score(y_test_tfidf, y_pred_tfidf)}")
    return lr_model

# Neural Network Model (LSTM)
def nn_model_training_evaluation(
    X_train_seq,
    X_test_seq,
    y_train_seq,
    y_test_seq,
    max_num_words,
    max_sequence_length,
    dropout,
    recurrent_dropout,
    output_dim):
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=max_num_words, output_dim=output_dim, input_length=max_sequence_length))
    model.add(LSTM(128, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_seq, y_train_seq, batch_size=256, epochs=3, validation_data=(X_test_seq, y_test_seq), verbose=1)

    # Evaluate the model
    score, acc = model.evaluate(X_test_seq, y_test_seq, batch_size=256)
    logger.info("\nLSTM Model Performance:")
    logger.info(f"Test accuracy: {acc}")
    return model


# Save the logistic regression model and TF-IDF vectorizer
def save_artifacts(lr_model, tfidf_vectorizer, nn_model, nn_tokenizer):
    joblib.dump(lr_model, 'lr_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    # Save the LSTM model and tokenizer
    nn_model.save('lstm_model.keras')
    joblib.dump(nn_tokenizer, 'tokenizer.pkl')


def init_models():
    ml_model_random_state = 42
    ml_max_features = 5000
    ml_test_size=0.2
    ml_max_iteration=1000

    nn_model_max_num_words = 5000
    nn_model_max_sequence_length = 100
    nn_max_test_size = 0.2
    nn_random_state = 42
    nn_dropout = 0.2
    nn_recurrent_dropout = 0.2
    nn_output_dim = 128

    if not config.force_regenerate_models and (
        os.path.exists('lr_model.pkl') and
        os.path.exists('lstm_model.keras') and
        os.path.exists('tfidf_vectorizer.pkl') and
        os.path.exists('tokenizer.pkl')):
            logger.info("Loading models, vectorizers, tokenizers from file")
            lr_model = joblib.load('lr_model.pkl')
            tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
            nn_lstm_model = keras.models.load_model('lstm_model.keras')
            nn_model_tokenizer = joblib.load('tokenizer.pkl')
    else:
        logger.info("Generating models")
        df = load_data()

        if config.enable_dev_test:
            df = df.sample(
                n=config.dev_test_sample_size,
                random_state=config.dev_test_random_state)

        # Display the first few rows
        logger.info("Dataset sample:")
        logger.info(df.head())

        preprocess_data(df)

        X_train_tfidf,X_test_tfidf, y_train_tfidf, y_test_tfidf, tfidf_vectorizer = ml_model_idf_vectorization(
            df,
            ml_max_features,
            ml_test_size,
            ml_model_random_state)

        lr_model = ml_model_training_evaluation(
            X_train_tfidf,
            X_test_tfidf,
            y_train_tfidf,
            y_test_tfidf,
            ml_max_iteration)

        X_train_seq, X_test_seq, y_train_seq, y_test_seq, nn_model_tokenizer = nn_model_tokenization_padding(
            df,
            nn_model_max_num_words,
            nn_model_max_sequence_length,
            nn_max_test_size,
            nn_random_state)

        nn_lstm_model = nn_model_training_evaluation(
            X_train_seq,
            X_test_seq,
            y_train_seq,
            y_test_seq,
            nn_model_max_num_words,
            nn_model_max_sequence_length,
            nn_dropout,
            nn_recurrent_dropout,
            nn_output_dim)

        save_artifacts(
            lr_model,
            tfidf_vectorizer,
            nn_lstm_model,
            nn_model_tokenizer)

    return lr_model, tfidf_vectorizer, nn_model_tokenizer, nn_lstm_model, nn_model_max_sequence_length


def sentiment_analysis_cli(lr_model, tfidf_vectorizer, tokenizer, nn_model, max_sequence_length):
    global logo
    print(f"{logo}")
    print("Type 'exit' to quit")
    while True:
        try:
            user_input = input("\nEnter a tweet or message: ")
            if user_input.lower() == 'exit':
                logger.info("Exiting...")
                break
            else:
                # Preprocess input
                clean_input = preprocess_text(user_input)

                # Logistic Regression Prediction
                input_tfidf = tfidf_vectorizer.transform([clean_input])
                lr_prob = lr_model.predict_proba(input_tfidf)[0][1]  # Probability of positive class
                lr_sentiment = 'Positive' if lr_prob >= 0.5 else 'Negative'

                # LSTM Prediction
                input_seq = tokenizer.texts_to_sequences([clean_input])
                input_seq = pad_sequences(input_seq, maxlen=max_sequence_length)
                lstm_prob = nn_model.predict(input_seq)[0][0]  # Probability from sigmoid output
                lstm_sentiment = 'Positive' if lstm_prob >= 0.5 else 'Negative'

                # Average the probabilities
                avg_prob = (lr_prob + lstm_prob) / 2
                overall_sentiment = 'Positive' if avg_prob >= 0.5 else 'Negative'

                # Display results
                print(f"\nLogistic Regression Sentiment: {lr_sentiment} (Confidence: {lr_prob:.2f})")
                print(f"LSTM Model Sentiment: {lstm_sentiment} (Confidence: {lstm_prob:.2f})")
                print(f"Overall Sentiment: {overall_sentiment} (Average Confidence: {avg_prob:.2f})")

        except Exception as e:
            logger.error(f"An error occurred: {e}")


def main():
    global logger
    logger = setup_logging()

    lr_model, tfidf_vectorizer, nn_model_tokenizer, nn_lstm_model, nn_model_max_sequence_length = init_models()

    sentiment_analysis_cli(
        lr_model,
        tfidf_vectorizer,
        nn_model_tokenizer,
        nn_lstm_model,
        nn_model_max_sequence_length)

# Run the CLI
if __name__ == '__main__':
    main()
