########################################
## Imports and setup
import os
import json
import psycopg2
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

## End imports
########################################


########################################
## Database setup
try:
    DATABASE_URL = os.environ['DATABASE_URL']
except:
    DATABASE_URL = 'sqlite:///predictions.db' 
    
db = connect(DATABASE_URL)

class BaseModel(Model):
    class Meta:
        database = db

class Prediction(BaseModel):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)
        
class Request(BaseModel):
    request = TextField()
    response = TextField()
    status = TextField()

def initialize_db():
    db.connect()
    db.create_tables([Prediction,Request], safe = True)
    
initialize_db() 

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
        error = "Field `observation` missing from request: {}".format(request)
        return False, error
    
    return True, ""

def check_request_id(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "id" not in request:
        error = "Field `id` missing from request: {}".format(request)
        return False, error
    
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
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

# End model un-pickling
########################################


########################################
# Begin webserver app

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    obs_dict = request.get_json()
  
    request_ok, error = check_request_id(obs_dict)
    if not request_ok:
        response = {'id': None,'error': error}
        r = Request(request = obs_dict, response = response, status = 'Error')
        r.save()
        return response

    _id = obs_dict['id']
    
    request_ok, error = check_request_observation(obs_dict)
    if not request_ok:
        response = {'id': _id,'error': error}
        r = Request(request = obs_dict, response = response, status = 'Error')
        r.save()
        return response
    
    observation = obs_dict['observation']
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    
    p = Prediction(observation_id=_id, proba=proba, observation=request.data)
    
    try:
        r = Request(request=obs_dict, response=response, status = 'Success')
        p.save()
        r.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        db.rollback()
        r = Request(request = obs_dict, response = response, status = 'Error')
        r.save()
        
    return jsonify(response)

# End webserver app
########################################

if __name__ == "__main__":
    app.run(debug=True)