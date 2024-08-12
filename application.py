from flask import Flask, render_template, request, jsonify
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
cors=CORS(app)

# Load the dataset and model
sugar = pd.read_csv("Diabetes.csv")
model = pickle.load(open('Diabetes_DT.pkl', 'rb'))


@app.route('/')
def index():
    # Prepare options for dropdowns
    Gender = sorted(sugar['gender'].unique())
    Smoking = sorted(sugar['smoking_history'].unique())
    Hypertension = sorted(sugar['hypertension'].unique())
    Heartproblem = sorted(sugar['heart_disease'].unique())
    Age_Group = sorted(sugar['age_group'].unique())
    BMI_group = sorted(sugar['bmi_group'].unique())
    Glucose = sorted(sugar['glucose_group'].unique())

    return render_template(
        "design.html",
        Gender=Gender,
        Smoking=Smoking,
        Hypertension=Hypertension,
        Heartproblem=Heartproblem,
        Age_Group=Age_Group,
        BMI_group=BMI_group,
        Glucose=Glucose
    )


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get form data
    Gender = request.form.get('Gender')
    Smoking = request.form.get('Smoking')
    Hypertension = request.form.get('Hypertension')
    Heartproblem = request.form.get('Heartproblem')
    Age_Group = request.form.get('Age_Group')
    BMI_group = request.form.get('BMI_group')
    Glucose = request.form.get('Glucose')
    HbA1c_level = request.form.get('HbA1c_level')

    # Create input DataFrame for the model
    input_data = pd.DataFrame(
        data=[[Gender, Smoking, Hypertension, Heartproblem, Age_Group, BMI_group, Glucose, HbA1c_level]],
        columns=['gender', 'smoking_history', 'hypertension', 'heart_disease', 'age_group', 'bmi_group',
                 'glucose_group', 'HbA1c_level'])

    # Predict using the model
    prediction = model.predict(input_data)
    print(jsonify({'prediction': int(prediction[0])}))


    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})


if __name__ == "__main__":
    app.run(debug = True)