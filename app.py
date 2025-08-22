from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model/credit_score_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    values = [
        float(data['age']),
        float(data['income']),
        float(data['salary']),
        float(data['bank']),
        float(data['card']),
        float(data['interest']),
        float(data['loan']),
        float(data['delay']),
        float(data['delayed']),
        float(data['inquiries']),
        float(data['debt']),
        float(data['ratio']),
        1 if data['min_payment'].lower() == 'yes' else 0,
        1 if data['behaviour'].lower() == 'good' else 0,
        float(data['balance'])
    ]
    prediction = model.predict([values])[0]
    prediction = round(prediction)
    return render_template('result.html', score=prediction)

if __name__ == '__main__':
    app.run(debug=True)
