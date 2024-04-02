from flask import Flask,request,jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/predictdata',methods=['POST'])
def predict_datapoint():
    try:
        payload = request.json
        data=CustomData(
            gender = payload.get("gender"),
            age = payload.get("age"),
            height = payload.get("height"),
            weight = payload.get("weight"),
            waist = payload.get("waist"),
            eyesight_left = payload.get("eyesight_left"),
            eyesight_right = payload.get("eyesight_right"),
            hearing_left = payload.get("hearing_left"),
            hearing_right = payload.get("hearing_right"),
            systolic = payload.get("systolic"),
            relaxation = payload.get("relaxation"),
            fasting_blood_sugar = payload.get("fasting_blood_sugar"),
            cholesterol = payload.get("cholesterol"),
            triglyceride = payload.get("triglyceride"),
            hdl = payload.get("hdl"),
            ldl = payload.get("ldl"),
            hemoglobin = payload.get("hemoglobin"),
            urine_protein = payload.get("urine_protein"),
            serum_creatinine = payload.get("serum_creatinine"),
            ast = payload.get("ast"),
            gtp = payload.get("gtp"),
            oral = payload.get("oral"),
            dental_caries = payload.get("dental_caries"),
            tartar = payload.get("tartar"),
            alt = payload.get("alt")
        )
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print("[app.py /predictdata]\n")
        print(results)
        return jsonify({"message": "success"}), 200
    except Exception as e:
        print(e)
        return jsonify({"message": "failure"}), 400
    
    
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port = 5001)        