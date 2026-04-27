from flask import Flask, render_template, request, jsonify
from inference import FakeNewsClassifier
import os

app = Flask(__name__)
classifier = FakeNewsClassifier()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        result = classifier.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 