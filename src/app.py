from flask import Flask, render_template, request
from predict_cli import SpamClassifier
import os

app = Flask(__name__)

# Configure template folder path
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
app.template_folder = template_dir

classifier = SpamClassifier()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form.get('message', '')
        if not message.strip():
            return render_template('index.html', error="Please enter a message")
            
        result = classifier.predict(message)
        return render_template('index.html', 
                            message=message,
                            result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)