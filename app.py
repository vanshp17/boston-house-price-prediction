import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash,session,escape
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open('linregmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    #standardization
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    result = model.predict(new_data)
    print(result[0])
    return jsonify(result[0])

if __name__ == '__main__':
    app.run(debug=True)