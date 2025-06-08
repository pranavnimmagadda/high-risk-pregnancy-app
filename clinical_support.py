import pandas as pd
import numpy as np
import pickle
import warnings
import re
import joblib

warnings.filterwarnings('always')

class HighRiskPregnancyClinicalSupport:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.model_threshold = 0.4
        self.feature_names = [
            'GRAVIDA', 'PARITY', 'ABORTIONS', 'HEIGHT', 'HEMOGLOBIN_mean',
            'age_adolescent', 'age_elderly', 'age_very_young', 'previous_loss',
            'recurrent_loss', 'gravida_parity_ratio', 'inadequate_anc', 'irregular_anc',
            'anemia_mild', 'anemia_moderate', 'anemia_severe', 'ever_severe_anemia',
            'systolic_bp', 'diastolic_bp', 'hypertension', 'BMI', 'underweight',
            'obese', 'normal_weight', 'depression', 'severe_depression', 'anxiety',
            'severe_anxiety', 'weight_gain', 'weight_gain_per_week', 'inadequate_weight_gain'
        ]

    def preprocess_patient(self, raw_data):
        try:
            required_keys = ['AGE', 'HEMOGLOBIN', 'HEMOGLOBIN_min', 'ABORTIONS', 'BP_last', 'GRAVIDA', 'PARITY',
                             'HEIGHT', 'WEIGHT_first', 'WEIGHT_last', 'NO_OF_WEEKS_max', 'TOTAL_ANC_VISITS',
                             'total_missed_visits', 'PHQ_SCORE_max', 'GAD_SCORE_max', 'MISSANC1FLG', 'MISSANC2FLG',
                             'MISSANC3FLG', 'MISSANC4FLG']
            for key in required_keys:
                if key not in raw_data:
                    raise KeyError(f"Missing required field: {key}")

            total_missed = sum([raw_data['MISSANC1FLG'], raw_data['MISSANC2FLG'], raw_data['MISSANC3FLG'], raw_data['MISSANC4FLG']])
            if raw_data['total_missed_visits'] != total_missed:
                raw_data['total_missed_visits'] = total_missed

            processed = {
                'age_adolescent': 1 if raw_data['AGE'] < 18 else 0,
                'age_very_young': 1 if raw_data['AGE'] < 16 else 0,
                'age_elderly': 1 if raw_data['AGE'] > 35 else 0,
                'GRAVIDA': raw_data['GRAVIDA'],
                'PARITY': raw_data['PARITY'],
                'ABORTIONS': raw_data['ABORTIONS'],
                'previous_loss': 1 if raw_data['ABORTIONS'] > 0 else 0,
                'recurrent_loss': 1 if raw_data['ABORTIONS'] >= 2 else 0,
                'gravida_parity_ratio': raw_data['GRAVIDA'] / raw_data['PARITY'] if raw_data['PARITY'] > 0 else raw_data['GRAVIDA'],
                'inadequate_anc': 1 if raw_data['TOTAL_ANC_VISITS'] < 4 else 0,
                'irregular_anc': 1 if raw_data['total_missed_visits'] >= 2 else 0,
                'HEMOGLOBIN_mean': raw_data['HEMOGLOBIN'],
                'anemia_mild': 1 if 10 <= raw_data['HEMOGLOBIN'] < 11 else 0,
                'anemia_moderate': 1 if 7 <= raw_data['HEMOGLOBIN'] < 10 else 0,
                'anemia_severe': 1 if raw_data['HEMOGLOBIN'] < 7 else 0,
                'ever_severe_anemia': 1 if raw_data['HEMOGLOBIN_min'] < 7 else 0,
                'HEIGHT': raw_data['HEIGHT'],
                'BMI': raw_data['WEIGHT_last'] / ((raw_data['HEIGHT'] / 100) ** 2) if raw_data['HEIGHT'] > 0 else 0,
                'underweight': 1 if (raw_data['WEIGHT_last'] / ((raw_data['HEIGHT'] / 100) ** 2) < 18.5) and raw_data['HEIGHT'] > 0 else 0,
                'obese': 1 if (raw_data['WEIGHT_last'] / ((raw_data['HEIGHT'] / 100) ** 2) > 30) and raw_data['HEIGHT'] > 0 else 0,
                'normal_weight': 1 if (18.5 <= raw_data['WEIGHT_last'] / ((raw_data['HEIGHT'] / 100) ** 2) <= 25) and raw_data['HEIGHT'] > 0 else 0,
                'depression': 1 if raw_data['PHQ_SCORE_max'] >= 10 else 0,
                'severe_depression': 1 if raw_data['PHQ_SCORE_max'] >= 15 else 0,
                'anxiety': 1 if raw_data['GAD_SCORE_max'] >= 10 else 0,
                'severe_anxiety': 1 if raw_data['GAD_SCORE_max'] >= 15 else 0,
                'weight_gain': raw_data['WEIGHT_last'] - raw_data['WEIGHT_first'],
                'weight_gain_per_week': (raw_data['WEIGHT_last'] - raw_data['WEIGHT_first']) / (raw_data['NO_OF_WEEKS_max'] or 1),
                'inadequate_weight_gain': 1 if ((raw_data['WEIGHT_last'] - raw_data['WEIGHT_first']) / (raw_data['NO_OF_WEEKS_max'] or 1)) < 0.2 else 0
            }

            bp_match = re.match(r'(\d+)/(\d+)', raw_data['BP_last'])
            if not bp_match:
                raise ValueError("Invalid BP_last format. Expected 'systolic/diastolic' (e.g., '120/80')")
            processed['systolic_bp'] = float(bp_match.group(1))
            processed['diastolic_bp'] = float(bp_match.group(2))
            processed['hypertension'] = 1 if processed['systolic_bp'] >= 140 or processed['diastolic_bp'] >= 90 else 0

            return pd.DataFrame([processed], columns=self.feature_names)
        except Exception as e:
            raise ValueError(f"Error in preprocessing: {str(e)}")

    def generate_high_risk_pregnancy_recommendations(self, raw_data):
        try:
            processed_data = self.preprocess_patient(raw_data)
            probability = self.model.predict_proba(processed_data)[:, 1][0]
            prediction = 1 if probability >= self.model_threshold else 0
            risk_level = self._categorize_high_risk(probability)

            recommendations = {
                'risk_assessment': {
                    'probability': float(probability * 100),  # Scale to percentage
                    'risk_level': risk_level,
                    'classification': bool(prediction)
                },
                'immediate_actions': [],
                'medication_protocols': [],
                'monitoring_protocols': {}
            }

            if risk_level == 'Critical High-Risk':
                recommendations['immediate_actions'].extend([
                    "🚨 CRITICAL HIGH-RISK PREGNANCY: Immediate specialized care required",
                    "Schedule emergency maternal-fetal medicine consultation within 24 hours",
                    "Consider hospitalization for comprehensive evaluation"
                ])
            elif risk_level == 'Very High-Risk':
                recommendations['immediate_actions'].extend([
                    "⚠️ VERY HIGH-RISK PREGNANCY: Enhanced monitoring required",
                    "Schedule high-risk pregnancy clinic appointment within 48 hours"
                ])
            elif risk_level == 'High-Risk':
                recommendations['immediate_actions'].extend([
                    "📋 HIGH-RISK PREGNANCY: Structured care plan required",
                    "Schedule high-risk pregnancy consultation within 1 week"
                ])
            else:
                recommendations['immediate_actions'].append("No immediate high-risk actions needed. Continue regular prenatal care.")

            if raw_data['HEMOGLOBIN'] < 7:
                recommendations['medication_protocols'].extend([
                    "🩸 SEVERE ANEMIA PROTOCOL:",
                    "• Iron sucrose IV 200mg in 100ml NS over 20 minutes",
                    "• Consider blood transfusion if symptomatic"
                ])
            elif raw_data['HEMOGLOBIN'] < 10:
                recommendations['medication_protocols'].extend([
                    "🩸 MODERATE ANEMIA MANAGEMENT:",
                    "• Ferrous sulfate 200mg TDS with Vitamin C",
                    "• Folic acid 5mg daily"
                ])
            if raw_data['ABORTIONS'] >= 2:
                recommendations['medication_protocols'].extend([
                    "🔄 RECURRENT LOSS PROTOCOL:",
                    "• Low-dose aspirin 75mg daily from 12 weeks",
                    "• Consider progesterone supplementation"
                ])
            if processed_data['hypertension'][0] == 1:
                recommendations['medication_protocols'].extend([
                    "🫀 HYPERTENSION MANAGEMENT:",
                    "• Methyldopa 250mg TDS (first line)",
                    "• Monitor blood pressure daily"
                ])
            if raw_data['PHQ_SCORE_max'] >= 10:
                recommendations['medication_protocols'].append("🧠 MENTAL HEALTH SUPPORT: Refer to mental health specialist for counseling")

            if risk_level in ['Critical High-Risk', 'Very High-Risk']:
                recommendations['monitoring_protocols'] = {
                    'anc_frequency': 'Weekly visits',
                    'fetal_monitoring': ['Biweekly non-stress tests (NST)', 'Weekly biophysical profile (BPP)'],
                    'maternal_monitoring': ['Daily blood pressure monitoring', 'Weekly hemoglobin checks']
                }
            else:
                recommendations['monitoring_protocols'] = {
                    'anc_frequency': 'Standard ANC schedule',
                    'fetal_monitoring': ['Standard fetal monitoring'],
                    'maternal_monitoring': ['Standard maternal monitoring']
                }

            return recommendations
        except Exception as e:
            raise ValueError(f"Error generating recommendations: {str(e)}")

    def _categorize_high_risk(self, probability):
        if probability >= 0.80:
            return 'Critical High-Risk'
        elif probability >= 0.60:
            return 'Very High-Risk'
        elif probability >= 0.40:
            return 'High-Risk'
        elif probability >= 0.20:
            return 'Moderate-Risk'
        else:
            return 'Low-Risk'

class prematureBirthClinicalSupport:
    def __init__(self, model_path):
        self.model_threshold = 0.6
        self.model = None
        self.feature_columns = [
            'GRAVIDA', 'AGE', 'PARITY', 'ABORTIONS', 'HEIGHT', 'HEMOGLOBIN_mean',
            'HEMOGLOBIN_min', 'HEMOGLOBIN_max', 'WEIGHT_anc_mean', 'WEIGHT_anc_min',
            'WEIGHT_anc_max', 'age_adolescent', 'age_elderly', 'age_very_young',
            'multigravida', 'grand_multipara', 'previous_loss', 'recurrent_loss',
            'gravida_parity_ratio', 'inadequate_anc', 'no_anc', 'irregular_anc',
            'missed_first_anc', 'consecutive_missed', 'anemia_mild', 'anemia_moderate',
            'anemia_severe', 'ever_severe_anemia', 'systolic_bp', 'diastolic_bp',
            'hypertension', 'severe_hypertension', 'BMI', 'underweight', 'obese',
            'normal_weight', 'depression', 'severe_depression', 'anxiety',
            'severe_anxiety', 'weight_gain', 'weight_gain_per_week',
            'inadequate_weight_gain'
        ]
        try:
            self.model = self._load_model(model_path)
        except Exception as e:
            warnings.warn(f"Failed to load model: {str(e)}. Using rule-based fallback.")

    def _load_model(self, model_path):
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")

    def _engineer_features(self, patient_data):
        try:
            required_keys = ['AGE', 'WEIGHT_last', 'WEIGHT_first', 'HEIGHT', 'HEMOGLOBIN', 'NO_OF_WEEKS_max',
                             'MISSANC1FLG', 'MISSANC2FLG', 'MISSANC3FLG', 'MISSANC4FLG', 'GRAVIDA', 'PARITY',
                             'ABORTIONS', 'TOTAL_ANC_VISITS', 'BP_last', 'PHQ_SCORE_max', 'GAD_SCORE_max']
            for key in required_keys:
                if key not in patient_data:
                    raise KeyError(f"Missing required field: {key}")

            processed = {
                'GRAVIDA': patient_data['GRAVIDA'],
                'AGE': patient_data['AGE'],
                'PARITY': patient_data['PARITY'],
                'ABORTIONS': patient_data['ABORTIONS'],
                'HEIGHT': patient_data['HEIGHT'],
                'HEMOGLOBIN_mean': patient_data['HEMOGLOBIN'],
                'HEMOGLOBIN_min': patient_data['HEMOGLOBIN'],
                'HEMOGLOBIN_max': patient_data['HEMOGLOBIN'],
                'WEIGHT_anc_mean': (patient_data['WEIGHT_last'] + patient_data['WEIGHT_first']) / 2,
                'WEIGHT_anc_min': patient_data['WEIGHT_first'],
                'WEIGHT_anc_max': patient_data['WEIGHT_last'],
                'age_adolescent': 1 if patient_data['AGE'] < 18 else 0,
                'age_very_young': 1 if patient_data['AGE'] < 16 else 0,
                'age_elderly': 1 if patient_data['AGE'] > 35 else 0,
                'multigravida': 1 if patient_data['GRAVIDA'] > 1 else 0,
                'grand_multipara': 1 if patient_data['PARITY'] > 5 else 0,
                'previous_loss': 1 if patient_data['ABORTIONS'] > 0 else 0,
                'recurrent_loss': 1 if patient_data['ABORTIONS'] >= 2 else 0,
                'gravida_parity_ratio': patient_data['GRAVIDA'] / (patient_data['PARITY'] + 1) if (patient_data['PARITY'] + 1) > 0 else 1,
                'inadequate_anc': 1 if patient_data['TOTAL_ANC_VISITS'] < 4 else 0,
                'no_anc': 1 if patient_data['TOTAL_ANC_VISITS'] == 0 else 0,
                'irregular_anc': 1 if (patient_data['MISSANC1FLG'] + patient_data['MISSANC2FLG'] + patient_data['MISSANC3FLG'] + patient_data['MISSANC4FLG']) >= 2 else 0,
                'missed_first_anc': patient_data['MISSANC1FLG'],
                'consecutive_missed': 1 if (patient_data['MISSANC1FLG'] + patient_data['MISSANC2FLG'] >= 2 or 
                                            patient_data['MISSANC2FLG'] + patient_data['MISSANC3FLG'] >= 2 or 
                                            patient_data['MISSANC3FLG'] + patient_data['MISSANC4FLG'] >= 2) else 0,
                'anemia_mild': 1 if 10 <= patient_data['HEMOGLOBIN'] < 11 else 0,
                'anemia_moderate': 1 if 7 <= patient_data['HEMOGLOBIN'] < 10 else 0,
                'anemia_severe': 1 if patient_data['HEMOGLOBIN'] < 7 else 0,
                'ever_severe_anemia': 1 if patient_data['HEMOGLOBIN'] < 7 else 0,
                'BMI': patient_data['WEIGHT_last'] / ((patient_data['HEIGHT'] / 100) ** 2) if patient_data['HEIGHT'] > 0 else 0,
                'underweight': 1 if (patient_data['WEIGHT_last'] / ((patient_data['HEIGHT'] / 100) ** 2) < 18.5) and patient_data['HEIGHT'] > 0 else 0,
                'obese': 1 if (patient_data['WEIGHT_last'] / ((patient_data['HEIGHT'] / 100) ** 2) > 30) and patient_data['HEIGHT'] > 0 else 0,
                'normal_weight': 1 if (18.5 <= patient_data['WEIGHT_last'] / ((patient_data['HEIGHT'] / 100) ** 2) <= 25) and patient_data['HEIGHT'] > 0 else 0,
                'depression': 1 if patient_data['PHQ_SCORE_max'] >= 10 else 0,
                'severe_depression': 1 if patient_data['PHQ_SCORE_max'] >= 15 else 0,
                'anxiety': 1 if patient_data['GAD_SCORE_max'] >= 10 else 0,
                'severe_anxiety': 1 if patient_data['GAD_SCORE_max'] >= 15 else 0,
                'weight_gain': patient_data['WEIGHT_last'] - patient_data['WEIGHT_first'],
                'weight_gain_per_week': (patient_data['WEIGHT_last'] - patient_data['WEIGHT_first']) / (patient_data['NO_OF_WEEKS_max'] or 1),
                'inadequate_weight_gain': 1 if ((patient_data['WEIGHT_last'] - patient_data['WEIGHT_first']) / (patient_data['NO_OF_WEEKS_max'] or 1)) < 0.2 else 0
            }

            bp_match = re.match(r'(\d+)/(\d+)', patient_data['BP_last'])
            if not bp_match:
                raise ValueError("Invalid BP_last format. Expected 'systolic/diastolic' (e.g., '120/80')")
            processed['systolic_bp'] = float(bp_match.group(1))
            processed['diastolic_bp'] = float(bp_match.group(2))
            processed['hypertension'] = 1 if processed['systolic_bp'] >= 140 or processed['diastolic_bp'] >= 90 else 0
            processed['severe_hypertension'] = 1 if processed['systolic_bp'] >= 160 or processed['diastolic_bp'] >= 110 else 0

            return pd.DataFrame([processed], columns=self.feature_columns)
        except Exception as e:
            raise ValueError(f"Error in preprocessing patient data: {str(e)}")

    def _rule_based_risk(self, patient_data):
        weight_last = patient_data.get('WEIGHT_last', 60)
        weight_first = patient_data.get('WEIGHT_first', 55)
        no_of_weeks_max = patient_data.get('NO_OF_WEEKS_max', 28)
        hemoglobin = patient_data.get('HEMOGLOBIN', 12)
        height = patient_data.get('HEIGHT', 160)
        age = patient_data.get('AGE', 25)
        missanc1flg = patient_data.get('MISSANC1FLG', 0)
        
        weight_gain = weight_last - weight_first
        weight_gain_per_week = weight_gain / no_of_weeks_max if no_of_weeks_max > 0 else 0
        bmi = weight_last / ((height / 100) ** 2) if height > 0 else 22
        
        risk_score = 0
        if weight_gain_per_week < 0.2:
            risk_score += 0.3
        if bmi < 18.5 or bmi > 30:
            risk_score += 0.2
        if height < 150:
            risk_score += 0.15
        if age < 20 or age > 35:
            risk_score += 0.1
        if hemoglobin < 11:
            risk_score += 0.15
        if missanc1flg == 1:
            risk_score += 0.1
        
        return min(risk_score, 1.0)

    def predict_risk(self, patient_data):
        if self.model:
            X = self._engineer_features(patient_data)
            try:
                return self.model.predict_proba(X)[:, 1][0]
            except Exception as e:
                warnings.warn(f"Model prediction failed: {str(e)}. Using rule-based fallback.")
        return self._rule_based_risk(patient_data)

    def generate_premature_birth_recommendations(self, patient_data):
        risk_probability = self.predict_risk(patient_data)  # Fractional probability (0.0 to 1.0)
        risk_level = self._categorize_premature_risk(risk_probability)
        classification = risk_probability >= self.model_threshold

        # Validate consistency between risk_level and classification
        if classification and risk_level in ['Low Premature Risk', 'Moderate Premature Risk']:
            warnings.warn(f"Inconsistent risk assessment: probability {risk_probability} classified as high but risk_level is {risk_level}. Adjusting risk_level.")
            risk_level = 'High Premature Risk'
        elif not classification and risk_level in ['High Premature Risk', 'Very High Premature Risk', 'Critical Premature Risk']:
            warnings.warn(f"Inconsistent risk assessment: probability {risk_probability} classified as low but risk_level is {risk_level}. Adjusting risk_level.")
            risk_level = 'Low Premature Risk'

        gestational_age = patient_data.get('NO_OF_WEEKS_max', 28)
        
        recommendations = {
            'risk_assessment': {
                'probability': float(risk_probability * 100),  # Scale to percentage for output
                'risk_level': risk_level,
                'classification': classification
            },
            'immediate_interventions': [],
            'premature_prevention_protocols': {},
            'fetal_maturation_interventions': [],
            'delivery_planning': {},
            'nicu_preparation': {},
            'medication_protocols': [],
            'monitoring_schedule': {},
            'healthcare_team_actions': {
                'asha_worker': [],
                'anm_nurse': [],
                'medical_officer': [],
                'obstetrician': [],
                'neonatologist': []
            }
        }
        
        self._add_premature_specific_actions(recommendations, patient_data, risk_probability, 
                                         risk_level, gestational_age)
        return recommendations

    def _categorize_premature_risk(self, probability):
        if probability >= 0.75:
            return 'Critical Premature Risk'
        elif probability >= 0.55:
            return 'Very High Premature Risk'
        elif probability >= 0.35:
            return 'High Premature Risk'
        elif probability >= 0.15:
            return 'Moderate Premature Risk'
        else:
            return 'Low Premature Risk'

    def _add_premature_specific_actions(self, recommendations, patient_data, probability, 
                                    risk_level, gestational_age):
        weight_last = patient_data.get('WEIGHT_last', 60)
        weight_first = patient_data.get('WEIGHT_first', 55)
        height = patient_data.get('HEIGHT', 160)
        hemoglobin = patient_data.get('HEMOGLOBIN', 12)
        missanc1flg = patient_data.get('MISSANC1FLG', 0)
        no_of_weeks_max = patient_data.get('NO_OF_WEEKS_max', 28)
        age = patient_data.get('AGE', 25)
        
        weight_gain = weight_last - weight_first
        weight_gain_per_week = weight_gain / no_of_weeks_max if no_of_weeks_max > 0 else 0
        inadequate_weight_gain = 1 if weight_gain_per_week < 0.2 else 0
        bmi = weight_last / ((height / 100) ** 2) if height > 0 else 22
        
        if risk_level == 'Critical Premature Risk':
            recommendations['immediate_interventions'].extend([
                "🚨 CRITICAL PREMATURE BIRTH RISK: Immediate intervention required",
                "Assess for signs of PREMATURE labor immediately",
                "Hospitalization for intensive monitoring",
                "Implement comprehensive PREMATURE prevention protocol",
                "Notify NICU team for potential PREMATURE delivery",
                "Prepare for emergency delivery if labor progresses"
            ])
        elif risk_level == 'Very High Premature Risk':
            recommendations['immediate_interventions'].extend([
                "⚠️ VERY HIGH PREMATURE RISK: Urgent prevention measures",
                "Initiate PREMATURE prevention protocol within 24 hours",
                "Urgent cervical length assessment via ultrasound",
                "Consider prophylactic interventions"
            ])
        elif risk_level == 'High Premature Risk':
            recommendations['immediate_interventions'].extend([
                "📋 HIGH Premature RISK: Enhanced surveillance required",
                "Implement Premature monitoring protocol",
                "Schedule frequent follow-up visits"
            ])
        
        self._add_premature_prevention_protocols(recommendations, patient_data, risk_level, gestational_age,
                                             inadequate_weight_gain, bmi, height, age, hemoglobin, missanc1flg)
        
        if risk_level in ['Critical Premature Risk', 'Very High Premature Risk'] and 24 <= gestational_age <= 34:
            recommendations['fetal_maturation_interventions'].extend([
                '💉 ANTENATAL CORTICOSTEROIDS:',
                '• Betamethasone 12mg IM x 2 doses, 24 hours apart',
                '• OR Dexamethasone 6mg IM q12h x 4 doses',
                '• Optimal benefit 24 hours to 7 days post-administration',
                '• Repeat course if high risk persists >14 days'
            ])
            
            if gestational_age < 32:
                recommendations['fetal_maturation_interventions'].extend([
                    '🧠 NEUROPROTECTION:',
                    '• Magnesium sulfate 4g IV bolus, then 1g/hr',
                    '• Continue until delivery or 24 hours',
                    '• For fetal neuroprotection against cerebral palsy'
                ])
        
        self._add_delivery_planning(recommendations, risk_level, gestational_age)
        
        if risk_level in ['Critical Premature Risk', 'Very High Premature Risk']:
            recommendations['nicu_preparation'] = {
                'nicu_notification': 'Immediate notification to NICU team',
                'bed_reservation': 'Reserve NICU bed for potential Premature delivery',
                'equipment_preparation': 'Prepare respiratory support and thermal regulation equipment',
                'staff_notification': 'Alert neonatal resuscitation team',
                'family_counseling': 'Counsel family on Premature birth outcomes and NICU expectations'
            }
        
        self._add_premature_medications(recommendations, patient_data, risk_level, gestational_age)
        self._add_premature_monitoring(recommendations, risk_level)
        self._add_premature_team_actions(recommendations, risk_level)

    def _add_premature_prevention_protocols(self, recommendations, patient_data, risk_level, gestational_age,
                                        inadequate_weight_gain, bmi, height, age, hemoglobin, missanc1flg):
        protocols = {}
        
        if inadequate_weight_gain == 1:
            protocols['nutritional_intervention'] = [
                'Urgent nutritional assessment and counseling',
                'High-calorie, high-protein diet plan (2500-3000 kcal/day)',
                'Target weight gain: 0.4-0.5kg/week in 2nd/3rd trimester',
                'Bi-weekly weight monitoring',
                'Prescribe prenatal vitamins and nutritional supplements'
            ]
        
        if bmi < 18.5 or bmi > 30:
            protocols['bmi_management'] = [
                'Detailed BMI assessment and monitoring',
                'For BMI <18.5: Nutritional supplementation to achieve healthy weight gain',
                'For BMI >30: Dietary counseling to manage weight gain',
                'Collaborate with dietitian for personalized plan'
            ]
        
        if height < 150:
            protocols['height_risk_management'] = [
                'Assess pelvic capacity due to short stature',
                'Monitor for cephalopelvic disproportion risk',
                'Consider early ultrasound for fetal size estimation'
            ]
        
        if age < 20 or age > 35:
            protocols['age_risk_management'] = [
                'Enhanced monitoring for adolescent (<20) or advanced maternal age (>35)',
                'Counsel on age-related premature risks',
                'Weekly assessments for age-related complications'
            ]
        
        if hemoglobin < 11.0:
            protocols['anemia_correction'] = [
                'Initiate aggressive iron therapy for hemoglobin <11g/dL',
                'Target hemoglobin >11g/dL to reduce premature risk',
                'Weekly hemoglobin monitoring during treatment',
                'Consider IV iron if oral intolerance or severe anemia'
            ]
        
        if missanc1flg == 1:
            protocols['enhanced_engagement'] = [
                'Intensive follow-up to ensure ANC compliance',
                'Weekly home visits by ASHA worker',
                'Address barriers to ANC attendance (transport, financial, cultural)',
                'Flexible scheduling for ANC appointments'
            ]
        
        if risk_level in ['High Premature Risk', 'Very High Premature Risk', 'Critical Premature Risk']:
            protocols['activity_modification'] = [
                'Restrict heavy lifting and strenuous physical activities',
                'Ensure 8-10 hours sleep and afternoon rest periods',
                'Implement stress reduction techniques (e.g., mindfulness)',
                'Avoid long-distance travel after 28 weeks',
                'Pelvic rest if cervical shortening detected'
            ]
            protocols['infection_prevention'] = [
                'Screen and treat urogenital infections promptly',
                'Educate on hygiene practices to prevent infections',
                'Avoid crowded places during infection seasons',
                'Treat bacterial vaginosis if present'
            ]
        
        recommendations['premature_prevention_protocols'] = protocols

    def _add_delivery_planning(self, recommendations, risk_level, gestational_age):
        if risk_level in ['Critical Premature Risk', 'Very High Premature Risk']:
            if gestational_age < 34:
                delivery_plan = {
                    'delivery_location': 'Tertiary center with Level III NICU',
                    'delivery_team': 'Obstetric team + Neonatal resuscitation team',
                    'anesthesia': 'Early anesthesia consultation for delivery planning',
                    'timing': 'Individualized based on maternal-fetal status',
                    'mode': 'Route of delivery based on obstetric indications'
                }
            else:
                delivery_plan = {
                    'delivery_location': 'Hospital with Level II nursery minimum',
                    'delivery_team': 'Standard obstetric team with neonatal support',
                    'anesthesia': 'Standard anesthesia consultation',
                    'timing': 'Aim for term delivery if stable',
                    'mode': 'Standard obstetric management'
                }
        else:
            delivery_plan = {
                'delivery_location': 'Standard delivery facility',
                'delivery_team': 'Standard obstetric care',
                'timing': 'Term delivery',
                'mode': 'Standard management'
            }
        recommendations['delivery_planning'] = delivery_plan

    def _add_premature_medications(self, recommendations, patient_data, risk_level, gestational_age):
        medications = []
        if risk_level == 'Critical Premature Risk' and 24 <= gestational_age <= 34:
            medications.extend([
                '🛑 TOCOLYTIC THERAPY (if in Premature labor):',
                '• Nifedipine 10mg sublingual, then 20mg oral q6h',
                '• OR Indomethacin 25mg q6h x 48 hours (if <32 weeks)',
                '• Limit tocolytics to 48 hours for steroid completion',
                '• Monitor maternal vital signs and fetal heart rate'
            ])
        if risk_level in ['High Premature Risk', 'Very High Premature Risk', 'Critical Premature Risk']:
            medications.extend([
                '🤰 PROGESTERONE SUPPLEMENTATION:',
                '• 17α-hydroxyprogesterone caproate 250mg IM weekly',
                '• Start 16-20 weeks, continue until 36 weeks',
                '• Indicated for history of spontaneous Premature birth',
                '• Monitor for injection site reactions'
            ])
        if risk_level in ['Very High Premature Risk', 'Critical Premature Risk']:
            medications.extend([
                '🪡 CERVICAL CERCLAGE CONSIDERATION:',
                '• Perform transvaginal ultrasound for cervical length',
                '• Consider cerclage if cervical length <25mm before 24 weeks',
                '• Administer prophylactic antibiotics during procedure',
                '.• Recommend modified activity post-cerclage'
            ])
        recommendations['medication_protocols'] = medications

    def _add_premature_monitoring(self, recommendations, risk_level):
        if risk_level == 'Critical Premature Risk':
            monitoring = {
                'visit_frequency': 'Weekly visits',
                'cervical_assessment': 'Cervical length every 1-2 weeks',
                'fetal_monitoring': 'Non-stress test (NST) twice weekly',
                'contraction_monitoring': 'Daily symptom assessment for contractions',
                'laboratory': 'Weekly CBC, CRP if infection suspected'
            }
        elif risk_level == 'Very High Premature Risk':
            monitoring = {
                'visit_frequency': 'Bi-weekly visits',
                'cervical_assessment': 'Cervical length every 2-3 weeks',
                'fetal_monitoring': 'Non-stress test (NST) weekly',
                'contraction_monitoring': 'Daily symptom diary',
                'laboratory': 'CBC every 2 weeks'
            }
        else:
            monitoring = {
                'visit_frequency': 'Standard ANC schedule',
                'cervical_assessment': 'Cervical length as indicated',
                'fetal_monitoring': 'Standard fetal monitoring',
                'contraction_monitoring': 'Educate on premature labor warning signs',
                'laboratory': 'Standard ANC labs'
            }
        recommendations['monitoring_schedule'] = monitoring

    def _add_premature_team_actions(self, recommendations, risk_level):
        if risk_level in ['Critical Premature Risk', 'Very High Premature Risk']:
            recommendations['healthcare_team_actions']['asha_worker'].extend([
                'Daily home visits during high-risk periods',
                'Monitor for premature labor symptoms (contractions, discharge)',
                'Ensure compliance with bed rest if prescribed',
                'Coordinate urgent transportation for labor signs',
                'Provide emotional support for premature risk anxiety'
            ])
            recommendations['healthcare_team_actions']['anm_nurse'].extend([
                'Assess for premature labor signs at each visit',
                'Monitor cervical changes if trained',
                'Educate on premature labor warning signs',
                'Coordinate urgent specialist referrals',
                'Administer tocolytic medications if prescribed'
            ])
            recommendations['healthcare_team_actions']['medical_officer'].extend([
                'Weekly premature labor risk assessments',
                'Perform cervical examinations as needed',
                'Coordinate referrals to tertiary care centers',
                'Manage premature labor emergencies',
                'Counsel family on premature risks and outcomes'
            ])
            recommendations['healthcare_team_actions']['obstetrician'].extend([
                'Comprehensive premature risk evaluation',
                'Assess need for cervical cerclage',
                'Manage tocolytic and steroid therapy',
                'Plan delivery timing and mode',
                'Coordinate with neonatology for high-risk cases'
            ])
            recommendations['healthcare_team_actions']['neonatologist'].extend([
                'Antenatal consultation for premature risk counseling',
                'Prepare NICU for potential premature delivery',
                'Attend delivery for premature births',
                'Plan ongoing care for premature infants'
            ])

def print_high_risk_pregnancy_recommendations(recommendations):
    output = ["🤰 HIGH-RISK PREGNANCY CLINICAL DECISION SUPPORT"]
    output.append("=" * 80)

    risk_assessment = recommendations['risk_assessment']
    output.append("\n📊 HIGH-RISK PREGNANCY ASSESSMENT:")
    output.append(f"   Risk Probability: {risk_assessment['probability']:.1f}%")
    output.append(f"   Risk Level: {risk_assessment['risk_level']}")
    output.append(f"   High-Risk Classification: {'YES' if risk_assessment['classification'] else 'NO'}")

    if recommendations['immediate_actions']:
        output.append("\n🚨 IMMEDIATE ACTIONS FOR HIGH-RISK PREGNANCY:")
        for i, action in enumerate(recommendations['immediate_actions'], 1):
            output.append(f"   {i}. {action}")

    if recommendations['medication_protocols']:
        output.append("\n💊 HIGH-RISK MEDICATION PROTOCOLS:")
        for protocol in recommendations['medication_protocols']:
            output.append(f"   {protocol}")

    if recommendations['monitoring_protocols']:
        output.append("\n📈 HIGH-RISK MONITORING PROTOCOLS:")
        monitoring = recommendations['monitoring_protocols']
        output.append(f"   • ANC Frequency: {monitoring.get('anc_frequency', 'Standard')}")
        if 'fetal_monitoring' in monitoring:
            output.append("   • Fetal Monitoring:")
            for item in monitoring['fetal_monitoring']:
                output.append(f"     - {item}")
        if 'maternal_monitoring' in monitoring:
            output.append("   • Maternal Monitoring:")
            for item in monitoring['maternal_monitoring']:
                output.append(f"     - {item}")

    output.append("\n" + "=" * 80)
    return "\n".join(output)

def print_premature_birth_recommendations(recommendations):
    output = ["👶 PREMATURE BIRTH PREVENTION CLINICAL DECISION SUPPORT"]
    output.append("="*80)

    risk_assessment = recommendations['risk_assessment']
    output.append("\n📊 PREMATURE BIRTH RISK ASSESSMENT:")
    output.append(f"   Risk Probability: {risk_assessment['probability']:.1f}%")
    output.append(f"   Risk Level: {risk_assessment['risk_level']}")
    output.append(f"   Premature Risk Classification: {'YES' if risk_assessment['classification'] else 'NO'}")

    if recommendations['immediate_interventions']:
        output.append("\n🚨 IMMEDIATE PREMATURE PREVENTION INTERVENTIONS:")
        for i, intervention in enumerate(recommendations['immediate_interventions'], 1):
            output.append(f"   {i}. {intervention}")

    if recommendations['premature_prevention_protocols']:
        output.append("\n🛡️ Premature BIRTH PREVENTION PROTOCOLS:")
        protocols = recommendations['premature_prevention_protocols']
        for protocol_name, actions in protocols.items():
            output.append(f"   • {protocol_name.replace('_', ' ').title()}:")
            for action in actions:
                output.append(f"     - {action}")

    if recommendations['fetal_maturation_interventions']:
        output.append("\n🧠 FETAL MATURATION INTERVENTIONS:")
        for intervention in recommendations['fetal_maturation_interventions']:
            output.append(f"   {intervention}")

    if recommendations['delivery_planning']:
        output.append("\n🏥 DELIVERY PLANNING:")
        delivery = recommendations['delivery_planning']
        output.append(f"   • Location: {delivery.get('delivery_location', 'Standard')}")
        output.append(f"   • Team: {delivery.get('delivery_team', 'Standard')}")
        output.append(f"   • Timing: {delivery.get('timing', 'Term')}")
        output.append(f"   • Mode: {delivery.get('mode', 'Standard')}")

    if recommendations['nicu_preparation']:
        output.append("\n🍼 NICU PREPARATION:")
        nicu = recommendations['nicu_preparation']
        for key, value in nicu.items():
            output.append(f"   • {key.replace('_', ' ').title()}: {value}")

    if recommendations['medication_protocols']:
        output.append("\n💊 PREMATURE PREVENTION MEDICATIONS:")
        for protocol in recommendations['medication_protocols']:
            output.append(f"   {protocol}")

    if recommendations['monitoring_schedule']:
        output.append("\n📅 MONITORING SCHEDULE:")
        monitoring = recommendations['monitoring_schedule']
        output.append(f"   • Visit Frequency: {monitoring.get('visit_frequency', 'Standard')}")
        output.append(f"   • Cervical Assessment: {monitoring.get('cervical_assessment', 'Standard')}")
        output.append(f"   • Fetal Monitoring: {monitoring.get('fetal_monitoring', 'Standard')}")
        output.append(f"   • Contraction Monitoring: {monitoring.get('contraction_monitoring', 'Standard')}")
        output.append(f"   • Laboratory: {monitoring.get('laboratory', 'Standard')}")

    if recommendations['healthcare_team_actions']:
        output.append("\n👥 HEALTHCARE TEAM ACTIONS:")
        team = recommendations['healthcare_team_actions']
        for role, actions in team.items():
            if actions:
                output.append(f"   • {role.replace('_', ' ').title()}:")
                for action in actions:
                    output.append(f"     - {action}")

    output.append("\n" + "=" * 80)
    return "\n".join(output)

if __name__ == "__main__":
    patient_data = {
        'AGE': 25,
        'WEIGHT_last': 64,
        'WEIGHT_first': 56,
        'HEIGHT': 160,
        'HEMOGLOBIN': 12.0,
        'NO_OF_WEEKS_max': 30,
        'MISSANC1FLG': 0,
        'MISSANC2FLG': 0,
        'MISSANC3FLG': 0,
        'MISSANC4FLG': 0,
        'GRAVIDA': 1,
        'PARITY': 0,
        'ABORTIONS': 0,
        'TOTAL_ANC_VISITS': 4,
        'BP_last': '118/76',
        'PHQ_SCORE_max': 2,
        'GAD_SCORE_max': 2
    }
    
    print("Starting clinical support execution...")
    clinical_support = prematureBirthClinicalSupport(model_path='best_model_fold3.pkl')
    print("Generating recommendations...")
    recommendations = clinical_support.generate_premature_birth_recommendations(patient_data)
    print("Recommendations generated. Printing now...")
    print(print_premature_birth_recommendations(recommendations))
    print("Execution complete.")