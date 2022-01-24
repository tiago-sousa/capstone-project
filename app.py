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
    
DB = connect(DATABASE_URL)

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

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
        return response

    _id = obs_dict['id']
    
    request_ok, error = check_request_observation(obs_dict)
    if not request_ok:
        response = {'id': _id,'error': error}
        return response 
    
    observation = obs_dict['observation']
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        DB.rollback()
    return jsonify(response)

# End webserver app
########################################

if __name__ == "__main__":
    app.run(debug=True)