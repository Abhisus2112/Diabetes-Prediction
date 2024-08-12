import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pickle

predictor = ['gender','smoking_history','hypertension','heart_disease','age_group','bmi_group','glucose_group','HbA1c_level']
target = ['diabetes']

df = pd.read_csv("Diabetes.csv")

X = df[predictor]
y = df[target]

trans1 = ColumnTransformer(transformers = [
    ('trans1',OneHotEncoder(sparse_output = False, drop = 'first'),['gender','smoking_history','age_group','bmi_group','glucose_group'])
],remainder = 'passthrough')

DT = DecisionTreeClassifier(max_depth = 3, criterion = 'gini')

pipe = make_pipeline(trans1,DT)

pipe.fit(X,y)

module = pickle.dump(pipe,open('Diabetes_DT.pkl','wb'))