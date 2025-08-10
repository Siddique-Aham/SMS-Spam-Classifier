import joblib
import re
import numpy as np
import os
from nltk.corpus import stopwords

class SpamClassifier:
    def __init__(self):
        # Get the absolute path to the models directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '../models/spam_classifier.pkl')
        vectorizer_path = os.path.join(script_dir, '../models/tfidf_vectorizer.pkl')
        
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")
        
        # Load models
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.stopwords = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stopwords])
        return text
    
    def analyze_message(self, message):
        # Check for spam indicators
        has_url = bool(re.search(r'http[s]?://|www\.', message.lower()))
        special_chars = len(re.findall(r'[^\w\s]', message))
        digit_count = len(re.findall(r'\d', message))
        
        return {
            'has_url': has_url,
            'special_chars': special_chars,
            'digit_count': digit_count
        }
    
    def predict(self, message):
        processed = self.preprocess_text(message)
        features = self.vectorizer.transform([processed])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Get top contributing words
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = features.toarray()[0]
        top_indices = np.argsort(tfidf_scores)[-5:][::-1]  # Top 5 features
        top_words = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
        
        # Analyze message
        analysis = self.analyze_message(message)
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': {
                'spam': float(probability[1]),
                'ham': float(probability[0])
            },
            'top_words': top_words,
            'analysis': analysis
        }

def main():
    try:
        classifier = SpamClassifier()
        
        print("SMS Spam Classifier - Command Line Interface")
        print("Type 'exit' to quit\n")
        
        while True:
            message = input("Enter an SMS message: ").strip()
            if message.lower() == 'exit':
                break
            if not message:
                print("Please enter a message\n")
                continue
                
            result = classifier.predict(message)
            
            print("\n=== Prediction Result ===")
            print(f"Message: {message}")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: Spam ({result['confidence']['spam']:.2%}), Ham ({result['confidence']['ham']:.2%})")
            
            print("\nReasons:")
            if result['top_words']:
                print("Top contributing words:")
                for word, score in result['top_words']:
                    print(f"- {word} (weight: {score:.2f})")
            
            if result['analysis']['has_url']:
                print("- Contains URL (common in spam)")
            if result['analysis']['special_chars'] > 2:
                print(f"- Contains {result['analysis']['special_chars']} special characters")
            if result['analysis']['digit_count'] > 5:
                print(f"- Contains {result['analysis']['digit_count']} digits (common in spam)")
            
            print("="*30 + "\n")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run train_model.py first")
        print("2. Verify these files exist:")
        print("   - models/spam_classifier.pkl")
        print("   - models/tfidf_vectorizer.pkl")
        print("3. Check file permissions")

if __name__ == '__main__':
    main()