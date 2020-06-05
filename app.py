# import flask
# import pickle
# import joblib

# model = joblib.load('finalized_model.pkl')

# app = flask.Flask(__name__, template_folder = 'templates')


# @app.route('/', methods = ['GET', 'POST'])

# def main():
#     if flask.request.method = 'GET':
#         return (flask.render_template('main.html'))
    
#     if flask.request.method = 'POST':
        
#         ID = flask.request.form ['ID']
#         OCCUPANCY_DATE = flask.request.form ['OCCUPANCY_DATE']
#         Existing Shelters In Area = flask.request.from ['Existing Shelters In Area']
#         Max Temp (C) = flask.request.form ['Max Temp (C)']
#         Min Temp (C) = flask.request.form ['Min Temp (C)']
#         ORGANIZATION_NAME = flask.request.form ['ORGANIZATION_NAME']
#         SHELTER_NAME = flask.requst.form ['SHELTER_NAME']
#         SHELTER_ADDRESS = flask.request.form ['SHELTER_ADDRESS']
#         SHELTER_CITY = flask.request.form['SHELTER_CITY']
#         SHELTER_PROVINCE = flask.request.from ['SHELTER_PROVINCE']
#         SHELTER_POSTAL_CODE = flask.request.form ['SHELTER_POSTAL_CODE']
#         FACILITY_NAME = flask.request.form ['FACILITY_NAME']
#         PROGRAM_NAME = flask.request.form['PROGRAM_NAME']
#         SECTOR = flask.request.form ['SECTOR']
# #         OCCUPANCY = flask.request.form ['OCCUPANCY']
#         CAPACITY = flask.request.forn['CAPACITY']
                                                             
        
#         input_variables = pd.DataFrame ([[ID,OCCUPANCY_DATE,Existing Shelters In Area,
#                                           Max Temp (C),Min Temp (C),ORGANIZATION_NAME,
#                                           SHELTER_NAME,SHELTER_ADDRESS,SHELTER_CITY,
#                                           SHELTER_PROVINCE,SHELTER_POSTAL_CODE,
#                                           FACILITY_NAME,PROGRAM_NAME,SECTOR,
#                                           OCCUPANCY,CAPACITY]], columns = ['ID','OCCUPANCY_DATE',
#                                                                                 'Existing Shelters In Area',
#                                                                                 'Max Temp (C)','Min Temp (C)',
#                                                                                 'ORGANIZATION_NAME',
#                                                                                 'SHELTER_NAME',
#                                                                                 'SHELTER_ADDRESS',
#                                                                                 'SHELTER_CITY',
#                                                                                 'SHELTER_PROVINCE',
#                                                                                 'SHELTER_POSTAL_CODE',
#                                                                                 'FACILITY_NAME',
#                                                                                 'PROGRAM_NAME' ,
#                                                                                 'SECTOR',
#                                                                                 'CAPACITY'])
#         prediction = model.predict (input_variables)[0]
#         return(flask.render_template('main.html', orginal_input = ))


# if __name__ == '__name__':
#     app.run()




    
# # cols = ['ID','OCCUPANCY_DATE','Existing Shelters In Area','Max Temp (C)','Min Temp (C)','ORGANIZATION_NAME','SHELTER_NAME','SHELTER_ADDRESS','SHELTER_CITY','SHELTER_PROVINCE','SHELTER_POSTAL_CODE','FACILITY_NAME','PROGRAM_NAME' ,'SECTOR' ,'OCCUPANCY' ,'CAPACITY']

    
    
    
    
    
    
    
    

from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

model = load_model('finalized_model')
# cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

cols = ['ID','OCCUPANCY_DATE','Existing Shelters In Area','Max Temp (C)','Min Temp (C)','ORGANIZATION_NAME','SHELTER_NAME','SHELTER_ADDRESS','SHELTER_CITY','SHELTER_PROVINCE','SHELTER_POSTAL_CODE','FACILITY_NAME','PROGRAM_NAME' ,'SECTOR' ,'CAPACITY']


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='The number of empty rooms is {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)



