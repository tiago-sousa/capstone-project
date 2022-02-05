from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]
    
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, mininum_records=250):
        super().__init__()
        self.mininum_records = mininum_records
            
    def fit(self, X, y=None):
        _X = X.copy()
        
        query_min = "count > "+ str(self.mininum_records)
        
        for col in _X.columns:
            if col == 'admission_type_code' :
                list_values = _X['admission_type_code'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.admission_type_code = [str(int(x)).lower() for x in list_values]
            elif col == 'discharge_disposition_code':
                list_values = _X['discharge_disposition_code'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.discharge_disposition_code = [str(int(x)).lower() for x in list_values]
            elif col == 'admission_source_code':
                list_values = _X['admission_source_code'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.admission_source_code = [str(int(x)).lower() for x in list_values]
            elif col == 'medical_specialty':
                list_values = _X['medical_specialty'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.medical_specialty = [str(x).lower() for x in list_values]
            elif col == 'payer_code':
                list_values = _X['payer_code'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.payer_code = [str(x).lower() for x in list_values]
            elif col == 'diag_1':
                list_values = _X['diag_1'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.diag_1 = [str(x).lower() for x in list_values]
            elif col == 'diag_2':
                list_values = _X['diag_2'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.diag_2 = [str(x).lower() for x in list_values]
            elif col == 'diag_3':
                list_values = _X['diag_3'].value_counts().reset_index(name="count").query(query_min)["index"]
                self.diag_3 = [str(x).lower() for x in list_values]

        return self

    def pre_process_text(self, obj):
        return str(obj).replace(" ","").lower()
    
    def bool_to_binary(self, obj):
        if obj == True:
            return 1
        elif obj == False:
            return 0
    
    def text_to_binary(self, obj):
        if obj == "yes" or obj == "ch" or obj=="1"  :
            return 1
        elif obj == "no" or obj =="0":
            return 0
        
    def handle_categories(self, obj, list_categories):
        if obj in list(list_categories):
            return str(obj)
        elif obj is None:
            return None
        else:
            return "others"
    
    def handle_missing_values(self, obj):
        if pd.isna(obj) or  str(obj).lower() in (None,"?","<na>","unknown/invalid","nan","none") :
            return None
        else :
            return obj
        
    def handle_invalid_categories(self, obj, invalid_categories):
        if obj in list(invalid_categories):
            return None
        else :
            return obj

    def is_float(self, obj):
        try:
            float(obj)
            return True
        except ValueError:
            return False
    
    def create_diag_category(self, obj):
        
        if not obj:
            return None
        elif str(obj)[:1].lower() in ("v","e"):
            return str(obj)[:1]
        elif self.is_float(obj):
            if float(obj) > 0 and float(obj) <= 139:
                return str('000-139')
            elif float(obj) >139  and float(obj) <= 239:
                return str('140-239')
            elif float(obj) > 239 and float(obj) <= 279:
                return str('240-279')
            elif float(obj) > 279 and float(obj) <= 289:
                return str('280-289')
            elif float(obj) > 289 and float(obj) <= 319:
                return str('290-319')
            elif float(obj) > 319 and float(obj) <= 389:
                return str('320-389')
            elif float(obj) > 389 and float(obj) <= 459:
                return str('390-459')
            elif float(obj) > 459 and float(obj) <= 519:
                return str('460-519')
            elif float(obj) > 519 and float(obj) <= 579:
                return str('520-579')
            elif float(obj) > 579 and float(obj) <= 629:
                return str('580-630')
            elif float(obj) > 629 and float(obj) <= 679:
                return str('630-679')
            elif float(obj) > 679 and float(obj) <= 709:
                return str('680-709')
            elif float(obj) > 709 and float(obj) <= 739:
                return str('710-739')         
            elif float(obj) > 739 and float(obj) <= 759:
                return str('740-759')
            elif float(obj) > 759 and float(obj) <= 779:
                return str('760-779')
            elif float(obj) > 779 and float(obj) <= 799:
                return str('780-799')
            elif float(obj) > 799 and float(obj) <= 999:
                return str('800-999')         
        else:
            return None
        
    def transform(self, X, y=None):
        _X = X.copy()
        
        #if 'diag_1' in _X.columns:
        #    values = _X['diag_1'].apply(self.pre_process_text)
        #    _X['diag_1_categories'] = values.apply(self.handle_missing_values)
        #    _X['diag_1_categories'] = _X['diag_1_categories'].apply(self.create_diag_category) 
        
        for _col in _X:     
            if _col in []:
                #_X[_col] = _X[_col].apply(self.bool_to_binary)
                _X[_col] = _X[_col].apply(self.text_to_binary)
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                
            elif _col in ['diuretics','insulin','change','diabetesMed','readmitted','has_prosthesis','blood_transfusion']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.text_to_binary)
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                
            elif _col in ['gender','age','weight','complete_vaccination_status','blood_type','max_glu_serum','A1Cresult']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                
            elif _col in ['race']:
                transformation = {"caucasian" : "caucasian", "white" : "caucasian", "european" : "caucasian", "euro" : "caucasian",
                  "africanamerican" : "afroamerican", "black" : "afroamerican", "afroamerican" : "afroamerican",
                  "latino" : "hispanic", "hispanic" : "hispanic", 
                  "asian":"asian", 
                  "?" : "missing", "other":"other"}                
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].map(transformation)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                
            elif _col in ['admission_type_code']:
                invalid_categories = ['5','6','8']
                _X[_col] = _X[_col].astype('Int64') 
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_invalid_categories, args =([invalid_categories]))
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.admission_type_code]))   
                
            elif _col in ['admission_source_code']:
                invalid_categories = ['9','15','17','20','21']
                _X[_col] = _X[_col].astype('Int64') 
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_invalid_categories, args =([invalid_categories]))
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.admission_source_code]))  
                
            elif _col in ['discharge_disposition_code']:
                invalid_categories = ['18','25','26']
                _X[_col] = _X[_col].astype('Int64') 
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_invalid_categories, args =([invalid_categories]))
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.discharge_disposition_code]))  
                
            elif _col in ['medical_specialty']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.medical_specialty]))  
                
            elif _col in ['payer_code']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.payer_code])) 
            
            elif _col in ['diag_1']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.diag_1])) 
                #_X[_col] = _X[_col].apply(self.create_diag_category) 
        
            elif _col in ['diag_2']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.diag_2])) 
        
            elif _col in ['diag_3']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
                _X[_col] = _X[_col].apply(self.handle_categories, args = ([self.diag_3])) 
        
        return _X
    
class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        _X = X.copy()
        for _col in _X:
            if _col in ['num_lab_procedures','num_procedures','num_medications','time_in_hospital','number_outpatient','number_emergency','number_inpatient','number_diagnoses']:
                _X[_col] = _X[_col].astype('Int64')
            elif _col in ['hemoglobin_level']:
                _X[_col] = _X[_col].astype('Float64')
        return _X

class SaveTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, step):
        self.step = step
        pass
    
    def fit(self, X=None, y=None, **fit_params):
        return self
    
    def transform(self, data):
        X = data.copy()
        name = "pipeline_"+self.step+"_spy.csv"
        pd.DataFrame(X).head(50).to_csv(name)
        return X