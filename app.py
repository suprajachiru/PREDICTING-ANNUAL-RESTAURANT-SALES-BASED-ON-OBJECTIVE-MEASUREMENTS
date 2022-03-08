from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import requests
from sklearn import *
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.preprocessing import LabelEncoder
# Create flask app
app = Flask(__name__,template_folder='templates')
@app.route("/", methods=['POST','GET'])

def Home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
	
    p2 = request.form.get("p2")
    p28 = request.form.get("p28")
    p6 = request.form.get("p6")
    Type_FC = request.form.get("Type_FC")
    P21 = request.form.get("P21")
    City_Group_Big = request.form.get("City Group_Big")   
    P29 = request.form.get("P29")
    P13 = request.form.get("P13")
    Type_IL = request.form.get("Type_IL")
    data = [p2,p28,p6,Type_FC,P21,City_Group_Big,P29,P13,Type_IL]
    df=pd.DataFrame(data)   
    minmax=MinMaxScaler()
    features=minmax.fit_transform(df)
    features=features.reshape(1,-1)
    stack_bal=pickle.load(open('model.pkl','rb'))
    prediction = stack_bal.predict(features)
    return render_template("index.html", prediction_revenue = prediction)
if __name__ == '__main__':
	app.run(debug=True)
