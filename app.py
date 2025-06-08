from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from clinical_support import HighRiskPregnancyClinicalSupport, prematureBirthClinicalSupport

app = Flask(__name__)
CORS(app)

try:
    high_risk_model = joblib.load('rf_model_fold2.pkl')
    premature_model = joblib.load('best_model_fold3.pkl')
    print("High Risk Model Features:", high_risk_model.feature_names_in_)
    print("Premature risk Model Features:", premature_model.feature_names_in_)
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    exit(1)

high_risk_support = HighRiskPregnancyClinicalSupport(model_path='rf_model_fold2.pkl')
premature_support = prematureBirthClinicalSupport(model_path='best_model_fold3.pkl')

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/predict_high_risk', methods=['POST'])
def predict_high_risk():
    try:
        data = request.json
        bp_last = data.get('BP_last', '120/80').strip()
        print(f"Received BP_last for high_risk: '{bp_last}'")
        if not isinstance(bp_last, str) or len(bp_last.split('/')) != 2:
            return jsonify({
                'status': 'error',
                'message': 'BP_last must be in the format "systolic/diastolic" (e.g., "120/80")'
            }), 400
        try:
            systolic, diastolic = map(int, bp_last.split('/'))
            if systolic <= 0 or diastolic <= 0:
                raise ValueError
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'BP_last values must be positive integers (e.g., "120/80")'
            }), 400

        # Prepare raw_data with all required keys
        raw_data = {
            'AGE': float(data.get('AGE', 0)),
            'HEMOGLOBIN': float(data.get('HEMOGLOBIN', 0)),
            'HEMOGLOBIN_min': float(data.get('HEMOGLOBIN_min', 0)),
            'ABORTIONS': int(data.get('ABORTIONS', 0)),
            'BP_last': bp_last,
            'GRAVIDA': int(data.get('GRAVIDA', 0)),
            'PARITY': int(data.get('PARITY', 0)),
            'HEIGHT': float(data.get('HEIGHT', 0)),
            'WEIGHT_first': float(data.get('WEIGHT_first', 0)),
            'WEIGHT_last': float(data.get('WEIGHT_last', 0)),
            'NO_OF_WEEKS_max': int(data.get('NO_OF_WEEKS_max', 0)),
            'TOTAL_ANC_VISITS': int(data.get('TOTAL_ANC_VISITS', 0)),
            'total_missed_visits': int(data.get('total_missed_visits', 0)),
            'PHQ_SCORE_max': int(data.get('PHQ_SCORE_max', 0)),
            'GAD_SCORE_max': int(data.get('GAD_SCORE_max', 0)),
            'MISSANC1FLG': int(data.get('MISSANC1FLG', 0)),
            'MISSANC2FLG': int(data.get('MISSANC2FLG', 0)),
            'MISSANC3FLG': int(data.get('MISSANC3FLG', 0)),
            'MISSANC4FLG': int(data.get('MISSANC4FLG', 0))
        }
        print("Input raw_data for high_risk:", raw_data)

        recommendations = high_risk_support.generate_high_risk_pregnancy_recommendations(raw_data)
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': f'Invalid input: {str(ve)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating recommendations: {str(e)}'
        }), 500

@app.route('/predict_premature', methods=['POST'])
def predict_premature():
    try:
        data = request.json
        bp_last = data.get('BP_last', '120/80').strip()
        print(f"Received BP_last for premature: '{bp_last}'")
        if not isinstance(bp_last, str) or len(bp_last.split('/')) != 2:
            return jsonify({
                'status': 'error',
                'message': 'BP_last must be in the format "systolic/diastolic" (e.g., "120/80")'
            }), 400
        try:
            systolic, diastolic = map(int, bp_last.split('/'))
            if systolic <= 0 or diastolic <= 0:
                raise ValueError
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'BP_last values must be positive integers (e.g., "120/80")'
            }), 400

        patient_data = {
            'AGE': float(data.get('AGE', 0)),
            'WEIGHT_last': float(data.get('WEIGHT_last', 0)),
            'WEIGHT_first': float(data.get('WEIGHT_first', 0)),
            'HEIGHT': float(data.get('HEIGHT', 0)),
            'HEMOGLOBIN': float(data.get('HEMOGLOBIN', 0)),
            'NO_OF_WEEKS_max': int(data.get('NO_OF_WEEKS_max', 0)),
            'MISSANC1FLG': int(data.get('MISSANC1FLG', 0)),
            'MISSANC2FLG': int(data.get('MISSANC2FLG', 0)),
            'MISSANC3FLG': int(data.get('MISSANC3FLG', 0)),
            'MISSANC4FLG': int(data.get('MISSANC4FLG', 0)),
            'GRAVIDA': int(data.get('GRAVIDA', 0)),
            'PARITY': int(data.get('PARITY', 0)),
            'ABORTIONS': int(data.get('ABORTIONS', 0)),
            'TOTAL_ANC_VISITS': int(data.get('TOTAL_ANC_VISITS', 0)),
            'BP_last': bp_last,
            'PHQ_SCORE_max': int(data.get('PHQ_SCORE_max', 0)),
            'GAD_SCORE_max': int(data.get('GAD_SCORE_max', 0))
        }
        print("Input patient_data for premature:", patient_data)

        recommendations = premature_support.generate_premature_birth_recommendations(patient_data)
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': f'Invalid input: {str(ve)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating recommendations: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)