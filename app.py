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
    observation_id = IntegerField(unique=True)
    observation = TextField()
    prediction = BooleanField()
    probability = FloatField()
    true_class = BooleanField(null=True)
    created_date = DateTimeField(default=datetime.datetime.now)
    modified_date = DateTimeField(null=True)
        
class Request(BaseModel):
    request = TextField()
    response = TextField()
    status = TextField()
    endpoint = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)
    
class Data(BaseModel):
    observation_id = IntegerField(unique=True)
    created_date = DateTimeField(default=datetime.datetime.now)
    modified_date = DateTimeField(null=True)
    admission_id = TextField(null=True)
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

def check_request_observation(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation" not in request:
        error_description = "Field `observation` missing from request: {}".format(request)
        return False, error_description
    
    return True, ""

def check_request_id(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "id" not in request:
        error_description = "Field `id` missing from request: {}".format(request)
        return False, error_description
    
    return True, ""

def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
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
                     'payer_code','medical_specialty',
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

def check_categorical_values(observation):
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
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        #else:
        #    error = "Categorical field {} missing"
        #    return False, error

    return True, ""

# End model un-pickling
########################################


########################################
# Begin webserver app

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    warning = False
    warning_description = ""
    
    obs_dict = request.get_json()

    request_ok, error_description = check_request_id(obs_dict)
    if not request_ok:
        response = {'id': None,'error': error_description}
        r = Request(request=obs_dict, response=response, endpoint='predict', status='error')
        r.save()
        return response

    _id = obs_dict['id']
    
    request_ok, error_description = check_request_observation(obs_dict)
    if not request_ok:
        response = {'id': _id,'error': error_description}
        r = Request(request=obs_dict, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    observation = obs_dict['observation']
    
    columns_ok, error_description, error_type = check_valid_column(observation)
    if not columns_ok and error_type=='failure':
        response = {'id': _id,'error': error_description}
        r = Request(request=obs_dict, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    if not request_ok and error_type=='warning':
        warning_description = error_description
        warning = True
    
    categories_ok, error_description = check_categorical_values(observation)
    if not categories_ok:
        response = {"observation_id": _id, 'error': error_description}
        r = Request(request=obs_dict, response=response, endpoint='predict', status='error')
        r.save()
        return jsonify(response)
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    probability = pipeline.predict_proba(obs)[0, 1]
    prediction = int(pipeline.predict(obs) [0])
    response = {'id':_id, 'probability': probability, 'prediction':prediction}
    
    p = Prediction(observation_id=_id, probability=probability, prediction=prediction, observation=observation)
    
    try:
        p.save()
        r = Request(request=obs_dict, response=response, endpoint='predict', status='success')
        if warning:
            response['warning'] = warning_description
        r.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        db.rollback()
        r = Request(request = obs_dict, response = response, endpoint = 'predict', status = 'error')
        r.save()
        
    return response

@app.route('/update', methods=['POST'])
def update():
    
    obs_dict = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs_dict['id'])
        p.true_class = obs_dict['true_class']
        p.modified_date = datetime.datetime.now()
        p.save()
        response = model_to_dict(p)
        r = Request(request=obs_dict, response=response, endpoint='update', status='success')
        r.save()
        return jsonify(response)
    
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs_dict['id'])
        response = {'error': error_msg}
        r = Request(request=obs_dict, response=response, endpoint='update', status='error')
        r.save()
        return jsonify(response)


# End webserver app
########################################

if __name__ == "__main__":
    app.run(debug=True)