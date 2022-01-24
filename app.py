# Initialization code DBN__
import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import BooleanField, Model, IntegerField, FloatField, TextField, IntegrityError
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from utils import custom_transformers

# Initialization code

########################################
# Begin database stuff

# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in predictions.db

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField()
    observation = TextField()
    pred = BooleanField()
    proba = FloatField()
    label = BooleanField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model    
          
with open('columns.json', 'r') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)
    
pipeline = joblib.load('pipeline.pickle')          
          
          
# End model un-pickling
########################################

########################################
# Unit Tests
########################################


########################################
# End Unit Tests
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():

    prediction = 0.5
    
    return jsonify({
        'prediction': prediction
    })
    
    

@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['prediction']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})
    
# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True)