from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('finalized_model')
# cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

cols = ['ID','OCCUPANCY_DATE','Existing Shelters In Area','Max Temp (\'b0C)','Min Temp (\'b0C)','ORGANIZATION_NAME\'92,\'92SHELTER_NAME\'92,\'91SHELTER_ADDRESS\'92,\'91SHELTER_CITY\'92,\'91SHELTER_PROVINCE\'92,\'91ORGANIZATION_NAME\'92,\'92SHELTER_POSTAL_CODE\'92,\'91FACILITY_NAME\'92,\'91SPROGRAM_NAME\'92 ,\'91SECTOR\'92 ,\'91SOCCUPANCY\'92 ,\'91CAPACITY\'92]\

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)



