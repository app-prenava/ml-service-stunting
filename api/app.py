import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models and configurations at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
try:
    model = joblib.load(os.path.join(MODEL_DIR, "stunting_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "stunting_threshold.pkl"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, feature_columns, threshold = None, None, 0.5

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not feature_columns:
        return jsonify({"status": "error", "message": "Model not loaded properly on server"}), 500

    try:
        data = request.get_json()
        
        # Calculate derived features (BMI)
        weight = float(data.get('mother_weight', 0))
        height = float(data.get('mother_height', 0))
        if height > 0:
            bmi = weight / ((height / 100) ** 2)
            if bmi < 10 or bmi > 60:
                bmi = np.nan
        else:
            bmi = np.nan
        
        # Build dataframe in the exact order of feature_columns
        input_dict = {
            'mother_height': data.get('mother_height'),
            'mother_weight': data.get('mother_weight'),
            'maternal_bmi': bmi,
            'maternal_hemoglobin': data.get('maternal_hemoglobin'),
            'mother_age': data.get('mother_age'),
            'mother_education': data.get('mother_education'),
            'birth_order': data.get('birth_order', 1), # Default 1 if not provided
            'anc_visits_total': data.get('anc_visits_total'),
            'iron_pills': data.get('iron_pills'),
            'pregnancy_sickness': data.get('pregnancy_sickness'),
            'birth_location_facility': data.get('birth_location_facility', 1), # Default facility
            'birth_weight': data.get('birth_weight'),
            'child_gender': data.get('child_gender'),
            'child_height': data.get('child_height'),
            'child_age_years': data.get('child_age_years', 0)
        }
        
        df = pd.DataFrame([input_dict])
        
        # Ensure all columns exist, fill missing with nan
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
                
        # Reorder to match model
        df = df[feature_columns]
        
        # Predict probability
        prob = model.predict_proba(df)[0, 1]
        
        # Apply optimal threshold
        pred_class = 1 if prob >= threshold else 0
        risk_label = "Tinggi" if pred_class == 1 else "Rendah"
        
        return jsonify({
            "status": "success",
            "prediction_class": pred_class,
            "stunting_risk": risk_label,
            "probability": round(float(prob), 4),
            "threshold_used": round(float(threshold), 4)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
