########################################
## Imports and setup
import os
import json
import psycopg2
import pickle
import joblib
import pandas as pd
import datetime
from flask import Flask, jsonify, request
from peewee import  *
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging

## End imports
########################################


########################################
## Database setup
try:
    DATABASE_URL = os.environ['DATABASE_URL']
except:
    DATABASE_URL = 'sqlite:///predictions.db' 
    
db = connect(DATABASE_URL)
db.connect()    

class BaseModel(Model):
    class Meta:
        database = db

class Prediction(BaseModel):
    admission_id = IntegerField(unique=True)
    observation = TextField()
    prediction = TextField()
    probability = FloatField()
    true_class = TextField(null=True)
    created_date = DateTimeField(default=datetime.datetime.now)
    modified_date = DateTimeField(null=True)
        
class Request(BaseModel):
    request = TextField()
    response = TextField()
    status = TextField()
    endpoint = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)
    
class Data(BaseModel):
    created_date = DateTimeField(default=datetime.datetime.now)
    modified_date = DateTimeField(null=True)
    admission_id = IntegerField(unique=True)
    patient_id = TextField(null=True)
    race = TextField(null=True)
    gender = TextField(null=True)
    age = TextField(null=True)
    weight = TextField(null=True)
    admission_type_code = TextField(null=True)
    discharge_disposition_code = TextField(null=True)
    admission_source_code = TextField(null=True)
    time_in_hospital = TextField(null=True)
    payer_code = TextField(null=True)
    medical_specialty = TextField(null=True)
    has_prosthesis = TextField(null=True)
    complete_vaccination_status = TextField(null=True)
    num_lab_procedures = TextField(null=True)
    num_procedures = TextField(null=True)
    num_medications = TextField(null=True)
    number_outpatient = TextField(null=True)
    number_emergency = TextField(null=True)
    number_inpatient = TextField(null=True)
    diag_1 = TextField(null=True)
    diag_2 = TextField(null=True)
    diag_3 = TextField(null=True)
    number_diagnoses = TextField(null=True)
    blood_type = TextField(null=True)
    hemoglobin_level = TextField(null=True)
    blood_transfusion = TextField(null=True)
    max_glu_serum = TextField(null=True)
    A1Cresult = TextField(null=True)
    diuretics = TextField(null=True)
    insulin = TextField(null=True)
    change = TextField(null=True)
    diabetesMed = TextField(null=True)
    readmitted = TextField(null=True)

db.create_tables([Prediction, Request, Data], safe = True)

# End database setup
########################################


########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')

# End model un-pickling
########################################


########################################
# Unpickle the previously-trained model

def check_valid_column(observation):
    
    valid_columns = {'admission_id',
                     'patient_id', 
                     'race', 
                     'gender', 
                     'age', 
                     'weight',
                     'admission_type_code', 
                     'discharge_disposition_code',
                     'admission_source_code', 
                     'time_in_hospital', 
                     'payer_code',
                     'medical_specialty',
                     'has_prosthesis',
                     'complete_vaccination_status',
                     'num_lab_procedures', 
                     'num_procedures', 
                     'num_medications',
                     'number_outpatient', 
                     'number_emergency', 
                     'number_inpatient', 
                     'diag_1',
                     'diag_2', 
                     'diag_3', 
                     'number_diagnoses', 
                     'blood_type',
                     'hemoglobin_level',
                     'blood_transfusion', 
                     'max_glu_serum', 
                     'A1Cresult',
                     'diuretics', 
                     'insulin', 
                     'change', 
                     'diabetesMed'
                    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error_description = "Missing columns: {}".format(missing)
        error_type = "failure"
        return False, error_description, error_type
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error_description = "Unrecognized columns provided: {}".format(extra)
        error_type = "warning"
        return False, error_description , error_type   

    return True, "",""

def check_column_types(observation):

    valid_column_types = {
                     'admission_id':[1,""],
                     'patient_id':[1,""], 
                     'race':["" , None], 
                     'gender':["" , None], 
                     'age':["" , None], 
                     'weight':["" , None],
                     'admission_type_code':[1.0,None], 
                     'discharge_disposition_code':[1.0,None],
                     'admission_source_code':[1,"",None], 
                     'time_in_hospital':[1,"",None], 
                     'payer_code':["",None],
                     'medical_specialty':["",None],
                     'has_prosthesis':[True,"",None],
                     'complete_vaccination_status':["",None],
                     'num_lab_procedures':[1.0,None], 
                     'num_procedures':[1,"",None],
                     'num_medications':[1.0,None], 
                     'number_outpatient':[1,"",None], 
                     'number_emergency':[1,"",None], 
                     'number_inpatient':[1,"",None], 
                     'diag_1':["",None], 
                     'diag_2':["",None], 
                     'diag_3':["",None], 
                     'number_diagnoses':[1,"",None], 
                     'blood_type':["",None], 
                     'hemoglobin_level':[1.0,None], 
                     'blood_transfusion':[True,"",None], 
                     'max_glu_serum':["",None], 
                     'A1Cresult':["",None], 
                     'diuretics':["",None], 
                     'insulin':["",None], 
                     'change':["",None], 
                     'diabetesMed':["",None], 
    }
    for key, valid_columns in valid_column_types.items():
        if key in observation:
            value = observation[key]
            if type(value).__name__ not in [type(x).__name__ for x in valid_columns]:
                error = "Invalid datatype provided for '{}': '{}'. Allowed datatypes are: {}".format(
                    key, type(value).__name__, ",".join(["'{}'".format(type(v).__name__) for v in valid_columns]))
                return False, error
            
            if key in ['admission_id','patient_id','admission_source_code','time_in_hospital','num_procedures','number_outpatient','number_emergency','number_inpatient','number_diagnoses'] and isinstance(value,str):
                try:
                    observation[key] = int(value)
                except:
                    error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( key, type(value).__name__,(type(1).__name__))
                    return False, error
                    
        

    return True, ""

def check_column_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        "age": [None,"","[70-80)","[60-70)","[50-60)","[80-90)","[40-50)","[30-40)","[90-100)","[20-30)","[10-20)","[0-10)"],
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format( key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        #else:
        #    error = "Categorical field {} missing"
        #    return False, error

    return True, ""


def get_model_prediction(pred_value):
    readmitted = ""
        
    if pred_value == 1:
        readmitted = "Yes"
    elif pred_value == 0 :
        readmitted = "No"
        
    return readmitted

# End model un-pickling
########################################


########################################
# Begin webserver app

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    warning = False
    warning_description = ""
    
    observation = request.get_json()
    
    columns_ok, error_description, error_type = check_valid_column(observation)
    if not columns_ok and error_type=='failure':
        response = {'admission_id': obs_dict['admission_id'],'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    if not columns_ok and error_type=='warning':
        warning_description = error_description
        warning = True
     
    _id = observation['admission_id']
    
    column_types_ok, error_description = check_column_types(observation)
    if not column_types_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    probability = pipeline.predict_proba(obs)[0, 1]
    prediction = get_model_prediction(pipeline.predict(obs)[0])
    response = {'readmitted':prediction}
    p = Prediction(admission_id=_id, probability=probability, prediction=prediction, observation=observation)
    
    try:
        p.save()
        r = Request(request=observation, response=response, endpoint='predict', status='success')
        if warning:
            response['warning'] = warning_description
        r.save()
    except IntegrityError:
        error_msg = "ERROR: Admission ID: '{}' already exists".format(_id)
        response = {'id':_id, 'error': error_msg}
        db.rollback()
        r = Request(request = observation, response = response, endpoint = 'predict', status = 'error')
        r.save()
        return response
        
    return response

@app.route('/update', methods=['POST'])
def update():
    
    observation = request.get_json()
    try:
        p = Prediction.get(Prediction.admission_id == observation['admission_id'])
        _id = observation['admission_id']
        p.true_class = observation['readmitted']
        p.modified_date = datetime.datetime.now()
        p.save()
        response = {'admission_id':_id, 'actual_readmitted':observation['readmitted'] , "predicted_readmitted":p.prediction }
        r = Request(request=observation, response=response, endpoint='update', status='success')
        r.save()
        return jsonify(response)
    
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(observation['admission_id'])
        response = {'error': error_msg}
        r = Request(request=observation, response=response, endpoint='update', status='error')
        r.save()
        return jsonify(response)


# End webserver app
########################################

if __name__ == "__main__":
    app.run(debug=True)