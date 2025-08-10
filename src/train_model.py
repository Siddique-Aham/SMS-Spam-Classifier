import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text data"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def load_data(filepath):
    """Load dataset in the specific format"""
    try:
        # Read CSV with correct format
        data = pd.read_csv(filepath, encoding='latin1', usecols=[0,1], names=['label', 'message'])
        
        # Clean the data
        data = data.dropna()
        data = data[data['message'].notnull()]
        data = data[data['label'].isin(['ham', 'spam'])]
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("\nYour file should have:")
        print("1. First column as label (ham/spam)")
        print("2. Second column as message text")
        print("3. May have extra empty columns after")
        exit(1)

def train_and_save_model():
    """Train model and save artifacts"""
    try:
        # Get paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '../data/sms_spam.csv')
        models_dir = os.path.join(script_dir, '../models')
        static_dir = os.path.join(script_dir, '../static')
        
        # Create directories if needed
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from: {data_path}")
        data = load_data(data_path)
        
        # Preprocess
        data['message'] = data['message'].apply(preprocess_text)
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        
        print("\nData loaded successfully!")
        print(f"Dataset size: {len(data)} messages")
        print(f"Spam count: {data['label'].sum()}")
        print(f"Ham count: {len(data) - data['label'].sum()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data['message'], data['label'], test_size=0.2, random_state=42
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_tfidf)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        model_path = os.path.join(models_dir, 'spam_classifier.pkl')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"\nModel saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plot_path = os.path.join(static_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion matrix plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your CSV has first column as ham/spam")
        print("2. Second column should contain message text")
        print("3. File should be in Latin1/UTF-8 encoding")
        print("4. No header row should be present")

if __name__ == '__main__':
    print("Starting SMS Spam Classifier Training...")
    train_and_save_model()