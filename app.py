from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizers
clf = joblib.load('model.pkl')
count_vect = joblib.load('count_vect.pkl')
tfidf_transformer = joblib.load('tfidf_transformer.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Looks for index.html inside templates folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    text_counts = count_vect.transform([message])
    text_tfidf = tfidf_transformer.transform(text_counts)
    pred = clf.predict(text_tfidf)
    label = 'Spam' if pred[0] == 1 else 'Not Spam'
    return jsonify({'prediction': label})

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

