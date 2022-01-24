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
    
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def pre_process_text(self, obj):
        return str(obj).replace(" ","").lower()
    
    def bool_to_binary(self, obj):
        if obj == True:
            return 1
        elif obj == False:
            return 0
    
    def text_to_binary(self, obj):
        if obj == "yes" or obj == "ch"  :
            return 1
        elif obj == "no":
            return 0
    
    def handle_missing_values(self, obj):
        if pd.isna(obj) or str(obj) == "?" or str(obj) == "unknown/invalid" or str(obj) == "nan" :
            return np.nan
        else :
            return obj
        
    def transform(self, X, y=None):
        _X = X.copy()
        for _col in _X:     
            if _col in ['has_prosthesis','blood_transfusion']:
                _X[_col] = _X[_col].apply(self.bool_to_binary)
            elif _col in ['diuretics','insulin','change','diabetesMed','readmitted']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.text_to_binary)
            elif _col in ['admission_source_code','discharge_disposition_code','admission_type_code','race','gender','age','weight','payer_code','medical_specialty','complete_vaccination_status','blood_type','max_glu_serum','A1Cresult','diag_1','diag_2','diag_3']:
                _X[_col] = _X[_col].apply(self.pre_process_text)
                _X[_col] = _X[_col].apply(self.handle_missing_values)
            
        return _X
    
class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def missing_to_zero(self, obj):
        if pd.isna(obj):
            return 0
        else :
            return obj
    
    def return_float(self, obj):
        return float(obj)
    
    def return_int(self, obj):
        return int(obj)
        
    def transform(self, X, y=None):
        _X = X.copy()
        for _col in _X:
            if _col in ['num_lab_procedures','num_procedures','num_medications']:
                _X[_col] = _X[_col].apply(self.missing_to_zero)
                _X[_col] = _X[_col].apply(self.return_int)
            elif _col in ['time_in_hospital','number_outpatient','number_emergency','number_inpatient','number_diagnoses']:
                _X[_col] = _X[_col].apply(self.return_int)
            elif _col in ['hemoglobin_level']:
                _X[_col] = _X[_col].apply(self.return_float)
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