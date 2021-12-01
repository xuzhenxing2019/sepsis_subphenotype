import shap
# shap.initjs()
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import time

def own_predict(X):
    threshold = 0.5
    X_test = X
    # C_0_result = classifier_0.predict_proba(X_test)[:, 1]
    # C_proba = C_0_result

    # C_1_result = classifier_1.predict_proba(X_test)[:, 1]
    # C_proba = C_1_result

    # C_2_result = classifier_2.predict_proba(X_test)[:, 1]
    # C_proba = C_2_result

    C_3_result = classifier_3.predict_proba(X_test)[:, 1]
    C_proba = C_3_result

# convert probability to label
    C_proba[C_proba >= threshold] = 1
    C_proba[C_proba < threshold] = 0

    y_predic = C_proba
    # predictions = pd.DataFrame(y_predic) #data=y_predic, columns=["column1"]

    return y_predic


file_path_classifier = '/Users/xuzhenxing/Documents/Sepsis/'
file_path_data = file_path_classifier
file_path_shap = '/Users/xuzhenxing/Documents/Sepsis/SHAP/'

# load the classifier
with open(file_path_classifier + 'classifier_0.pkl', 'rb') as fid_0:
    classifier_0 = pkl.load(fid_0)

with open(file_path_classifier + 'classifier_1.pkl', 'rb') as fid_1:
    classifier_1 = pkl.load(fid_1)

with open(file_path_classifier + 'classifier_2.pkl', 'rb') as fid_2:
    classifier_2 = pkl.load(fid_2)

with open(file_path_classifier + 'classifier_3.pkl', 'rb') as fid_3:
    classifier_3 = pkl.load(fid_3)

# load data
data_df = pd.read_csv(file_path_data+'data_df.csv',index_col=0)

feature_cols = ['SOFA_score', 'Respiration_score', 'Coagulation_score', 'Liver_score', 'Cardiovascular_score',
                'CNS_score', 'Renal_score',
                'Bands', 'CRP', 'Temperature', 'WBC', 'SO2', 'Pao2', 'Respiratory_rate', 'Bicarbonate', 'Heart_rate',
                'Lactate', 'Systolic_ABP',
                'Troponin I', 'BUN', 'Creatinine', 'ALT', 'AST', 'Bilirubin', 'GCS','Hemoglobin', 'INR', 'Platelet',
                'Albumin', 'Chloride', 'Glucose',
                'Sodium', 'RDW', 'Lymphocyte_count', 'Lymphocyte_percent', 'BMI', 'Age', 'Comorbidity_score'] #,

X = data_df[feature_cols]
Y = data_df['group'] # obtaining label

X_test = X  # X[:10]

# reset for testing
class_flag = '0' # ['0','1','2','3']
feature_order = False # True,False
sample_flag = 10


start = pd.Timestamp.now()

rf_explainer = shap.KernelExplainer(own_predict, data= shap.sample(X_test,sample_flag), task='classification')
rf_shap_values = rf_explainer.shap_values(X_test)
# print('rf_shap_values',rf_shap_values)

save_shap_value = pd.DataFrame(rf_shap_values, columns=feature_cols)
save_shap_value.to_csv(file_path_shap+ 'feature_importance_class_'+ class_flag +'_sample_flag_'+str(sample_flag)+'_shap_value.csv')


print('the compute time:',pd.Timestamp.now()-start)

fig = plt.figure(figsize=(25,10))
# shap.summary_plot(rf_shap_values, X_test, max_display=X_test.shape[1],sort=feature_order)
shap.summary_plot(rf_shap_values, X_test, max_display=X_test.shape[1])

fig_name = file_path_shap+ 'feature_importance_class_'+ class_flag +'_sample_flag_'+str(sample_flag)+'.pdf'
fig.savefig(fig_name)
plt.close('all')





