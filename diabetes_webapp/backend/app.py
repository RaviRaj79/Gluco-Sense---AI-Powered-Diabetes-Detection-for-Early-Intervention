from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load
import os
import numpy as np  # Import NumPy

app = Flask(__name__)
CORS(app)

# Construct the absolute path to the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'XGBoost_Diabetes_Model.joblib')

try:
    model = load(model_path)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        print("Received data:", data)

        required_fields = ['gender', 'smoking_history', 'age', 'bmi',
                           'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        try:
            age = float(data['age'])
            bmi = float(data['bmi'])
            HbA1c_level = float(data['HbA1c_level'])
            blood_glucose_level = float(data['blood_glucose_level'])
            hypertension = int(data['hypertension'])
            heart_disease = int(data['heart_disease'])

        except ValueError as e:
            return jsonify({'error': f"Invalid numerical input: {e}"}), 400

        input_data = pd.DataFrame([{
            'gender': data['gender'],
            'smoking_history': data['smoking_history'],
            'age': age,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'hypertension': hypertension,
            'heart_disease': heart_disease
        }])

        try:
            input_data = input_data[model.named_steps['preprocessor'].feature_names_in_]
        except KeyError as e:
            return jsonify({'error': f'Column mismatch: {e}'}), 400

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[:, 1][0]

        risk_score_percentage = round(probability * 100, 2)

        if risk_score_percentage < 30:
            risk_level = "Low Risk"
        elif 30 <= risk_score_percentage < 70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        bmi_risk = ''
        hba1c_risk = ''
        glucose_risk = ''
        hypertension_risk = ''
        heart_disease_risk = ''
        health_advice = []

        try:
            if bmi < 18.5:
                bmi_risk = "Underweight (Moderate Risk)"
            elif 18.5 <= bmi < 24.9:
                bmi_risk = "Normal (Low Risk)"
            elif 25 <= bmi < 29.9:
                bmi_risk = "Overweight (Moderate Risk)"
            else:
                bmi_risk = "Obese (High Risk)"
                health_advice.append("Consider a healthy diet and exercise to reduce obesity-related risks.")
        except Exception as e:
            return jsonify({'error': f"BMI Calculation Error: {e}"}), 400

        try:
            if HbA1c_level < 5.7:
                hba1c_risk = "Normal (Low Risk)"
            elif 5.7 <= HbA1c_level < 6.5:
                hba1c_risk = "Prediabetes (Moderate Risk)"
                health_advice.append("Monitor your diet and exercise to prevent diabetes.")
            else:
                hba1c_risk = "Diabetic (High Risk)"
                health_advice.append("Consult a doctor to manage high blood sugar levels.")
        except Exception as e:
            return jsonify({'error': f"HbA1c Level Assessment Error: {e}"}), 400

        try:
            if blood_glucose_level < 140:
                glucose_risk = "Normal (Low Risk)"
            elif 140 <= blood_glucose_level < 199:
                glucose_risk = "Prediabetes (Moderate Risk)"
                health_advice.append("Control sugar intake and maintain an active lifestyle.")
            else:
                glucose_risk = "High (High Risk)"
                health_advice.append("High blood glucose detected. Seek medical guidance.")
        except Exception as e:
            return jsonify({'error': f"Blood Glucose Level Assessment Error: {e}"}), 400

        try:
            hypertension_risk = "Yes (High Risk)" if hypertension == 1 else "No (Low Risk)"
            if hypertension == 1:
                health_advice.append("Manage blood pressure with a healthy lifestyle and checkups.")
        except Exception as e:
            return jsonify({'error': f"Hypertension Risk Assessment Error: {e}"}), 400

        try:
            heart_disease_risk = "Yes (High Risk)" if heart_disease == 1 else "No (Low Risk)"
            if heart_disease == 1:
                health_advice.append("Heart disease detected. Follow a heart-healthy lifestyle.")
        except Exception as e:
            return jsonify({'error': f"Heart Disease Risk Assessment Error: {e}"}), 400

        if data['smoking_history'] in ['current', 'former']:
            health_advice.append("Quit smoking to lower your risk of diabetes and cardiovascular diseases.")

        # Convert NumPy floats to standard Python floats
        results = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_score_percentage': float(risk_score_percentage),  # Convert to float
            'risk_level': risk_level,
            'bmi_risk': bmi_risk,
            'hba1c_risk': hba1c_risk,
            'glucose_risk': glucose_risk,
            'hypertension_risk': hypertension_risk,
            'heart_disease_risk': heart_disease_risk,
            'health_advice': health_advice
        }

        return jsonify(results)

    except Exception as e:
        error_message = str(e)
        print(f"Prediction error: {error_message}")
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)