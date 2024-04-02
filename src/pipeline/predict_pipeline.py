import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            data_scaled=preprocessor.transform(features)

            print(len(data_scaled[0]))
            print(data_scaled)
            preds=model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, 
                 gender,
                 age,
                 height,
                 weight,
                 waist,
                 eyesight_left,
                 eyesight_right,
                 hearing_left,
                 hearing_right,
                 systolic,
                 relaxation,
                 fasting_blood_sugar,
                 cholesterol,
                 triglyceride,
                 hdl,
                 ldl,
                 hemoglobin,
                 urine_protein,
                 serum_creatinine,
                 ast,
                 alt,
                 gtp,
                 oral,
                 dental_caries,
                 tartar
                 ):

        self.gender = gender
        self.age = age
        self.height = height
        self.weight = weight
        self.waist = waist
        self.eyesight_left = eyesight_left
        self.eyesight_right = eyesight_right
        self.hearing_left = hearing_left
        self.hearing_right = hearing_right
        self.systolic = systolic
        self.relaxation = relaxation
        self.fasting_blood_sugar = fasting_blood_sugar
        self.cholesterol = cholesterol
        self.triglyceride = triglyceride
        self.hdl = hdl
        self.ldl = ldl
        self.hemoglobin = hemoglobin
        self.urine_protein = urine_protein
        self.serum_creatinine = serum_creatinine
        self.ast = ast
        self.gtp = gtp
        self.oral = oral
        self.dental_caries = dental_caries
        self.tartar = tartar
        self.alt = alt

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "height": [self.height],
                "weight": [self.weight],
                "waist": [self.waist],
                "eyesight_left": [self.eyesight_left],
                "eyesight_right": [self.eyesight_right],
                "hearing_left": [self.hearing_left],
                "hearing_right": [self.hearing_right],
                "systolic": [self.systolic],
                "relaxation": [self.relaxation],
                "fasting_blood_sugar": [self.fasting_blood_sugar],
                "cholesterol": [self.cholesterol],
                "triglyceride": [self.triglyceride],
                "hdl": [self.hdl],
                "ldl": [self.ldl],
                "hemoglobin": [self.hemoglobin],
                "urine_protein": [self.urine_protein],
                "serum_creatinine": [self.serum_creatinine],
                "ast": [self.ast],
                "gtp": [self.gtp],
                "oral": [self.oral],
                "dental_caries": [self.dental_caries],
                "tartar": [self.tartar],
                "alt": [self.alt]
            }

            x = pd.DataFrame(custom_data_input_dict)
            
            return x

        except Exception as e:
            raise CustomException(e, sys)