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
    patient_id = IntegerField(null=True)
    race = TextField(null=True)
    gender = TextField(null=True)
    age = TextField(null=True)
    weight = TextField(null=True)
    admission_type_code = FloatField(null=True)
    discharge_disposition_code = FloatField(null=True)
    admission_source_code = IntegerField(null=True)
    time_in_hospital = IntegerField(null=True)
    payer_code = TextField(null=True)
    medical_specialty = TextField(null=True)
    has_prosthesis = BooleanField(null=True)
    complete_vaccination_status = TextField(null=True)
    num_lab_procedures = FloatField(null=True)
    num_procedures = IntegerField(null=True)
    num_medications = FloatField(null=True)
    number_outpatient = IntegerField(null=True)
    number_emergency = IntegerField(null=True)
    number_inpatient = IntegerField(null=True)
    diag_1 = TextField(null=True)
    diag_2 = TextField(null=True)
    diag_3 = TextField(null=True)
    number_diagnoses = IntegerField(null=True)
    blood_type = TextField(null=True)
    hemoglobin_level = FloatField(null=True)
    blood_transfusion = BooleanField(null=True)
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

def check_request_id(request):
    
    if "admission_id" not in request:
        error_description = "Field `admission_id` missing from request: {}".format(request)
        return False, error_description

    return True, ""

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
                     'admission_id':[1, 1.0],
                     'patient_id':[1, 1.0], 
                     'race':["" , None], 
                     'gender':["" , None], 
                     'age':["" , None], 
                     'weight':["" , None],
                     'admission_type_code':[1.0,1, None], 
                     'discharge_disposition_code':[1.0, 1,None],
                     'admission_source_code':[1.0, 1,None], 
                     'time_in_hospital':[1,1.0, None], 
                     'payer_code':["",None],
                     'medical_specialty':["",None],
                     'has_prosthesis':[True,1, 1.0, None],
                     'complete_vaccination_status':["",None],
                     'num_lab_procedures':[1.0,1,None], 
                     'num_procedures':[1,1.0, None],
                     'num_medications':[1,1.0, None],
                     'number_outpatient':[1,1.0, None],
                     'number_emergency':[1,1.0, None],
                     'number_inpatient':[1,1.0, None],
                     'diag_1':["",None], 
                     'diag_2':["",None], 
                     'diag_3':["",None], 
                     'number_diagnoses':[1,1.0, None],
                     'blood_type':["",None], 
                     'hemoglobin_level':[1.0,1,None], 
                     'blood_transfusion':[True,1,1.0, None], 
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
                    
    return True, ""


def check_admission_id(observation):
    value = observation['admission_id']
    
    if type(value).__name__ != type(1).__name__:
        if value.is_integer():
            observation['admission_id'] = int(observation['admission_id'])
        else:
            error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'admission_id', type(observation['admission_id']).__name__,(type(1).__name__))
            return False, error      
    
    return True, ''

def check_patient_id(observation):
    value = observation['patient_id']
    
    if type(value).__name__ != type(1).__name__:
        if value.is_integer():
            observation['patient_id'] = int(observation['patient_id'])
        else:
            error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'patient_id', type(observation['patient_id']).__name__,(type(1).__name__))
            return False, error      
    
    return True, ''

def check_age(observation):
    
    valid_values = [None,"","?","[70-80)","[60-70)","[50-60)","[80-90)","[40-50)","[30-40)","[90-100)","[20-30)","[10-20)","[0-10)"]
    
    if observation['age']:
        if observation['age'].strip() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'age' , observation['age'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      

        observation['age'] = observation['age'].strip()
    
    return True, ''

def check_weight(observation):
    valid_values = [None,"","?","[75-100)","[50-75)","[100-125)","[125-150)","[25-50)","[0-25)","[150-175)","[175-200)",">200"]
    
    if observation['weight']:
        if observation['weight'].strip() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'weight' , observation['weight'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      

        observation['weight'] = observation['weight'].strip()
    
    return True, ''


def check_gender(observation):
    valid_values = [None,"","?","male","female","unknown/invalid"]
    
    if observation['gender']:
        if observation['gender'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'gender' , observation['gender'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['gender'] = observation['gender'].strip().lower()
    
    return True, ''


def check_admission_type_code(observation):

    if observation['admission_type_code']:
        if type(observation['admission_type_code']).__name__ == type(1).__name__:
            observation['admission_type_code'] = float(observation['admission_type_code'])
        elif type(observation['admission_type_code']).__name__ == type(1.0).__name__:
            if observation['admission_type_code'].is_integer():
                observation['admission_type_code'] = float(observation['admission_type_code'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'admission_type_code', type(observation['admission_type_code']).__name__,(type(1).__name__))
                return False, error
        else:
            error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'admission_type_code', type(observation['admission_type_code']).__name__,(type(1).__name__))
            return False, error
    
    return True, ''

def check_discharge_disposition_code(observation):
    
    if observation['discharge_disposition_code']:
        if type(observation['discharge_disposition_code']).__name__ == type(1).__name__:
            observation['discharge_disposition_code'] = float(observation['discharge_disposition_code'])
        elif type(observation['discharge_disposition_code']).__name__ == type(1.0).__name__:
            if observation['discharge_disposition_code'].is_integer():
                observation['discharge_disposition_code'] = float(observation['discharge_disposition_code'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'discharge_disposition_code', type(observation['discharge_disposition_code']).__name__,(type(1).__name__))
                return False, error
        else:
            error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'discharge_disposition_code', type(observation['discharge_disposition_code']).__name__,(type(1).__name__))
            return False, error
    
    return True, ''


def check_admission_source_code(observation):
    
    if observation['admission_source_code']:
        if type(observation['admission_source_code']).__name__ == type(1).__name__:
            observation['admission_source_code'] = float(observation['admission_source_code'])
        elif type(observation['admission_source_code']).__name__ == type(1.0).__name__:
            if observation['admission_source_code'].is_integer():
                observation['admission_source_code'] = float(observation['admission_source_code'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'admission_source_code', type(observation['admission_source_code']).__name__,(type(1).__name__))
                return False, error
        else:
            error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'admission_source_code', type(observation['admission_source_code']).__name__,(type(1).__name__))
            return False, error
    return True, ''

def check_time_in_hospital(observation):
    
    if observation['time_in_hospital']:
        if type(observation['time_in_hospital']).__name__ == type(1).__name__:
            observation['time_in_hospital'] = float(observation['time_in_hospital'])
        elif type(observation['time_in_hospital']).__name__ == type(1.0).__name__:
            if observation['time_in_hospital'].is_integer():
                observation['time_in_hospital'] = float(observation['time_in_hospital'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'time_in_hospital', type(observation['time_in_hospital']).__name__,(type(1).__name__))
                return False, error
        else:
            error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'time_in_hospital', type(observation['time_in_hospital']).__name__,(type(1).__name__))
            return False, error
        if observation['time_in_hospital']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'time_in_hospital', observation['time_in_hospital'])
            return False, error
    
    return True, ''

def check_complete_vaccination_status(observation):
    valid_values = [None,"none","complete","incomplete"]
    
    if observation['complete_vaccination_status']:
        if observation['complete_vaccination_status'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'complete_vaccination_status' , observation['complete_vaccination_status'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['complete_vaccination_status'] = observation['complete_vaccination_status'].strip().lower()
    
    return True, ''

def check_num_lab_procedures(observation):

    if observation['num_lab_procedures']:
        if type(observation['num_lab_procedures']).__name__ == type(1).__name__:
            observation['num_lab_procedures'] = int(observation['num_lab_procedures'])
        
        elif type(observation['num_lab_procedures']).__name__ == type(1.0).__name__:
            if observation['num_lab_procedures'].is_integer():
                observation['num_lab_procedures'] = int(observation['num_lab_procedures'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'num_lab_procedures', type(observation['num_lab_procedures']).__name__,(type(1).__name__))
                return False, error
    
        if observation['num_lab_procedures']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'num_lab_procedures', observation['num_lab_procedures'])
            return False, error
    return True, ''

def check_num_procedures(observation):

    if observation['num_procedures']:
        if type(observation['num_procedures']).__name__ == type(1).__name__:
            observation['num_procedures'] = int(observation['num_procedures'])
        
        elif type(observation['num_procedures']).__name__ == type(1.0).__name__:
            if observation['num_procedures'].is_integer():
                observation['num_procedures'] = int(observation['num_procedures'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'num_procedures', type(observation['num_procedures']).__name__,(type(1).__name__))
                return False, error
    
        if observation['num_procedures']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'num_procedures', observation['num_procedures'])
            return False, error
    return True, ''

def check_num_medications(observation):

    if observation['num_medications']:
        if type(observation['num_medications']).__name__ == type(1).__name__:
            observation['num_medications'] = int(observation['num_medications'])
        
        elif type(observation['num_medications']).__name__ == type(1.0).__name__:
            if observation['num_medications'].is_integer():
                observation['num_medications'] = int(observation['num_medications'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'num_medications', type(observation['num_medications']).__name__,(type(1).__name__))
                return False, error
    
        if observation['num_medications']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'num_medications', observation['num_medications'])
            return False, error
    return True, ''

def check_number_outpatient(observation):

    if observation['number_outpatient']:
        if type(observation['number_outpatient']).__name__ == type(1).__name__:
            observation['number_outpatient'] = int(observation['number_outpatient'])
        
        elif type(observation['number_outpatient']).__name__ == type(1.0).__name__:
            if observation['number_outpatient'].is_integer():
                observation['number_outpatient'] = int(observation['number_outpatient'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'number_outpatient', type(observation['number_outpatient']).__name__,(type(1).__name__))
                return False, error
    
        if observation['number_outpatient']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'number_outpatient', observation['number_outpatient'])
            return False, error
    return True, ''

def check_number_emergency(observation):

    if observation['number_emergency']:
        if type(observation['number_emergency']).__name__ == type(1).__name__:
            observation['number_emergency'] = int(observation['number_emergency'])
        
        elif type(observation['number_emergency']).__name__ == type(1.0).__name__:
            if observation['number_emergency'].is_integer():
                observation['number_emergency'] = int(observation['number_emergency'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'number_emergency', type(observation['number_emergency']).__name__,(type(1).__name__))
                return False, error
    
        if observation['number_emergency']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'number_emergency', observation['number_emergency'])
            return False, error
    return True, ''

def check_number_inpatient(observation):

    if observation['number_inpatient']:
        if type(observation['number_inpatient']).__name__ == type(1).__name__:
            observation['number_inpatient'] = int(observation['number_inpatient'])
        
        elif type(observation['number_inpatient']).__name__ == type(1.0).__name__:
            if observation['number_inpatient'].is_integer():
                observation['number_inpatient'] = int(observation['number_inpatient'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'number_inpatient', type(observation['number_inpatient']).__name__,(type(1).__name__))
                return False, error
    
        if observation['number_inpatient']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'number_inpatient', observation['number_inpatient'])
            return False, error
    return True, ''


def check_number_diagnoses(observation):

    if observation['number_diagnoses']:
        if type(observation['number_diagnoses']).__name__ == type(1).__name__:
            observation['number_diagnoses'] = int(observation['number_diagnoses'])
        
        elif type(observation['number_diagnoses']).__name__ == type(1.0).__name__:
            if observation['number_diagnoses'].is_integer():
                observation['number_diagnoses'] = int(observation['number_diagnoses'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'number_diagnoses', type(observation['number_diagnoses']).__name__,(type(1).__name__))
                return False, error
    
        if observation['number_diagnoses']<0:
            error = "Invalid value provided for '{}': '{}'. Value cannot be negative".format( 'number_inpatient', observation['number_diagnoses'])
            return False, error
    return True, ''

def check_blood_type(observation):
    valid_values = [None,"","?","o+","a+","b+","o-","a-","ab+","b-","ab-"]
    
    if observation['blood_type']:
        if observation['blood_type'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'blood_type' , observation['blood_type'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['blood_type'] = observation['blood_type'].strip().lower()
    
    return True, ''
    
def check_hemoglobin_level(observation):
    if observation['hemoglobin_level']:
        if observation['hemoglobin_level']<0 or observation['hemoglobin_level']>100:
            error = "Invalid value provided for '{}': '{}'. Value outside expected range".format( 'hemoglobin_level', observation['hemoglobin_level'])
            return False, error
    return True, ''

def check_A1Cresult(observation):
    valid_values = [None,"none","norm",">8",">7"]
    
    if observation['A1Cresult']:
        if observation['A1Cresult'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'A1Cresult' , observation['A1Cresult'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['A1Cresult'] = observation['A1Cresult'].strip().lower()
    
    return True, ''

def check_max_glu_serum(observation):
    valid_values = [None,"none","norm",">200",">300"]
    
    if observation['max_glu_serum']:
        if observation['max_glu_serum'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'max_glu_serum' , observation['max_glu_serum'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['max_glu_serum'] = observation['max_glu_serum'].strip().lower()
    
    return True, ''

def check_diuretics(observation):
    valid_values = [None,"yes","no"]
    
    if observation['diuretics']:
        if observation['diuretics'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'diuretics' , observation['diuretics'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['diuretics'] = observation['diuretics'].strip().lower()
    
    return True, ''

def check_insulin(observation):
    valid_values = [None,"yes","no"]
    
    if observation['insulin']:
        if observation['insulin'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'insulin' , observation['insulin'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['insulin'] = observation['insulin'].strip().lower()
    
    return True, ''

def check_diabetesMed(observation):
    valid_values = [None,"yes","no"]
    
    if observation['diabetesMed']:
        if observation['diabetesMed'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'diabetesMed' , observation['diabetesMed'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['diabetesMed'] = observation['diabetesMed'].strip().lower()
    
    return True, ''

def check_change(observation):
    valid_values = [None,"ch","no"]
    
    if observation['change']:
        if observation['change'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'change' , observation['change'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
        observation['change'] = observation['change'].strip().lower()
    
    return True, ''

def check_update_requests(observation):
    valid_columns = {'admission_id',
                     'readmitted'
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

def check_has_prosthesis(observation):
    
    if observation['has_prosthesis']:
        if type(observation['has_prosthesis']).__name__ == type(1).__name__:
            observation['has_prosthesis'] = int(observation['has_prosthesis'])
        elif type(observation['has_prosthesis']).__name__ == type(1.0).__name__:
            if observation['has_prosthesis'].is_integer():
                observation['has_prosthesis'] = int(observation['has_prosthesis'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'admission_source_code', type(observation['admission_source_code']).__name__,(type(1).__name__))
                return False, error
        if observation['has_prosthesis'] in (True,1):
            observation['has_prosthesis'] = "1"
        elif observation['has_prosthesis'] in (False,0):
            observation['has_prosthesis'] = "0"
        else:    
            error = "Invalid value provided for '{}': '{}'".format( 'has_prosthesis' , observation['has_prosthesis'])
            return False, error      
    return True, ''


def check_blood_transfusion(observation):
    
    if observation['blood_transfusion']:
        if type(observation['blood_transfusion']).__name__ == type(1).__name__:
            observation['blood_transfusion'] = int(observation['blood_transfusion'])
        elif type(observation['blood_transfusion']).__name__ == type(1.0).__name__:
            if observation['blood_transfusion'].is_integer():
                observation['blood_transfusion'] = int(observation['blood_transfusion'])
            else:
                error = "Invalid datatype provided for '{}': '{}'. Transformation to '{}' is not possible".format( 'blood_transfusion', type(observation['blood_transfusion']).__name__,(type(1).__name__))
                return False, error
        if observation['blood_transfusion'] in (True,1):
            observation['blood_transfusion'] = "1"
        elif observation['blood_transfusion'] in (False,0):
            observation['blood_transfusion'] = "0"
        else:    
            error = "Invalid value provided for '{}': '{}'".format( 'blood_transfusion' , observation['blood_transfusion'])
            return False, error      
    return True, ''


def check_column_types_update(observation):

    valid_column_types = {
                     'admission_id':[1],
                     'readmitted':[""], 
    }
    for key, valid_columns in valid_column_types.items():
        if key in observation:
            value = observation[key]
            if type(value).__name__ not in [type(x).__name__ for x in valid_columns]:
                error = "Invalid datatype provided for '{}': '{}'. Allowed datatypes are: {}".format(
                    key, type(value).__name__, ",".join(["'{}'".format(type(v).__name__) for v in valid_columns]))
                return False, error
                    
    return True, ""

def check_readmitted(observation):
    valid_values = ["yes","no"]
    
    if observation['readmitted']:
        if observation['readmitted'].strip().lower() not in valid_values:
            error = "Invalid value provided for '{}': '{}'. Allowed values are: {}".format( 'readmitted' , observation['readmitted'], ",".join(["'{}'".format(v) for v in valid_values]))
            return False, error      
    
    return True, ''

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
    
    request_ok, error_description = check_request_id(observation)
    if not request_ok:
        response = {'id': None,'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response

    columns_ok, error_description, error_type = check_valid_column(observation)
    if not columns_ok and error_type=='failure':
        response = {'admission_id': observation['admission_id'],'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    if not columns_ok and error_type=='warning':
        warning_description = error_description
        warning = True
    
    column_types_ok, error_description = check_column_types(observation)
    if not column_types_ok:
        response = {"admission_id": observation['admission_id'], 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    admission_id_ok, error_description = check_admission_id(observation)
    if not admission_id_ok:
        response = {"admission_id": observation['admission_id'], 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    _id = observation['admission_id']
    
    patient_id_ok, error_description = check_patient_id(observation)
    if not patient_id_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    """age_ok, error_description = check_age(observation)
    if not age_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    weight_ok, error_description = check_weight(observation)
    if not weight_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    gender_ok, error_description = check_gender(observation)
    if not gender_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    """
    admission_type_code_ok, error_description = check_admission_type_code(observation)
    if not admission_type_code_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response

    admission_source_code_ok, error_description = check_admission_source_code(observation)
    if not admission_source_code_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    discharge_disposition_code_ok, error_description = check_discharge_disposition_code(observation)
    if not discharge_disposition_code_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    check_time_in_hospital_ok, error_description = check_time_in_hospital(observation)
    if not check_time_in_hospital_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response 
    
    check_complete_vaccination_status_ok, error_description = check_complete_vaccination_status(observation)
    if not check_complete_vaccination_status_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_num_lab_procedures_ok, error_description = check_num_lab_procedures(observation)
    if not check_num_lab_procedures_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_num_medications_ok, error_description = check_num_medications(observation)
    if not check_num_medications_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response  
    
    check_num_procedures_ok, error_description = check_num_procedures(observation)
    if not check_num_procedures_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_number_outpatient_ok, error_description = check_number_outpatient(observation)
    if not check_number_outpatient_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_number_emergency_ok, error_description = check_number_emergency(observation)
    if not check_number_emergency_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_number_inpatient_ok, error_description = check_number_inpatient(observation)
    if not check_number_inpatient_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_number_diagnoses_ok, error_description = check_number_diagnoses(observation)
    if not check_number_diagnoses_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_blood_type_ok, error_description = check_blood_type(observation)
    if not check_blood_type_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_hemoglobin_level_ok, error_description = check_hemoglobin_level(observation)
    if not check_hemoglobin_level_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response   
    
    check_max_glu_serum_ok, error_description = check_max_glu_serum(observation)
    if not check_max_glu_serum_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response
    
    check_A1Cresult_ok, error_description = check_A1Cresult(observation)
    if not check_A1Cresult_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response      
    
    check_diuretics_ok, error_description = check_diuretics(observation)
    if not check_diuretics_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response 
    
    check_insulin_ok, error_description = check_insulin(observation)
    if not check_insulin_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    check_diabetesMed_ok, error_description = check_diabetesMed(observation)
    if not check_diabetesMed_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    check_change_ok, error_description = check_change(observation)
    if not check_change_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response    
    
    check_has_prosthesis_ok, error_description = check_has_prosthesis(observation)
    if not check_has_prosthesis_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='predict', status='error')
        r.save()
        return response  

    check_blood_transfusion_ok, error_description = check_blood_transfusion(observation)
    if not check_blood_transfusion_ok:
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
        d = Data( admission_id = observation['admission_id'], patient_id = observation['patient_id'], race = observation['race'], gender = observation['gender'], age = observation['age'],weight = observation['weight'], admission_type_code = observation['admission_type_code'], discharge_disposition_code = observation['discharge_disposition_code'], admission_source_code = observation['admission_source_code'], time_in_hospital = observation['time_in_hospital'], payer_code = observation['payer_code'], medical_specialty = observation['medical_specialty'], has_prosthesis = observation['has_prosthesis'], complete_vaccination_status = observation['complete_vaccination_status'], num_lab_procedures = observation['num_lab_procedures'], num_procedures = observation['num_procedures'], num_medications = observation['num_medications'], number_outpatient = observation['number_outpatient'], number_emergency = observation['number_emergency'], number_inpatient = observation['number_inpatient'], diag_1 = observation['diag_1'], diag_2 = observation['diag_2'], diag_3 = observation['diag_3'], number_diagnoses = observation['number_diagnoses'], blood_type = observation['blood_type'], hemoglobin_level = observation['hemoglobin_level'], blood_transfusion = observation['blood_transfusion'], max_glu_serum = observation['max_glu_serum'], A1Cresult = observation['A1Cresult'], diuretics = observation['diuretics'], insulin = observation['insulin'], change = observation['change'], diabetesMed = observation['diabetesMed'])
        d.save()
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
    
    warning = False
    warning_description = ""
    
    observation = request.get_json()
    
    request_ok, error_description = check_request_id(observation)
    if not request_ok:
        response = {'id': None,'error': error_description}
        r = Request(request=observation, response=response, endpoint='update', status='error')
        r.save()
        return response

    _id = observation['admission_id']    

    columns_ok, error_description, error_type = check_update_requests(observation)
    if not columns_ok  and error_type=='failure':
        response = {'id': None,'error': error_description}
        r = Request(request=observation, response=response, endpoint='update', status='error')
        r.save()
        return response
    
    if not columns_ok and error_type=='warning':
        warning_description = error_description
        warning = True
    
    check_column_types_update_ok, error_description = check_column_types_update(observation)
    if not check_column_types_update_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='update', status='error')
        r.save()
        return response
    
    check_readmitted_ok, error_description = check_readmitted(observation)
    if not check_readmitted_ok:
        response = {"admission_id": _id, 'error': error_description}
        r = Request(request=observation, response=response, endpoint='update', status='error')
        r.save()
        return response    
  
    try:
        p = Prediction.get(Prediction.admission_id == _id)
        p.true_class = observation['readmitted']
        p.modified_date = datetime.datetime.now()
        p.save()
        response = {'admission_id':_id, 'actual_readmitted':observation['readmitted'] , "predicted_readmitted":p.prediction }
        if warning:
            response['warning'] = warning_description
        r = Request(request=observation, response=response, endpoint='update', status='success')
        r.save()
        d = Data.get(Data.admission_id == _id)
        d.modified_date = datetime.datetime.now()
        d.readmitted = observation['readmitted']
        d.save()
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