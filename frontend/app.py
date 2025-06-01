from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model from parent directory's 'models' folder
model = joblib.load('../models/fraud_detection_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input features V1 to V28
            features = [float(request.form[f'v{i}']) for i in range(1, 29)]

            # Add Amount and Time to features
            amount = float(request.form['Amount'])
            time = float(request.form['Time'])
            features.extend([amount, time])

            # Predict using trained model
            prediction = model.predict([features])[0]

            if prediction == 1:
                prediction = "🚨 Fraudulent Transaction"
            else:
                prediction = "✅ Legitimate Transaction"
        except Exception as e:
            prediction = f"❌ Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
