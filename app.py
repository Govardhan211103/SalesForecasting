from flask import Flask,render_template, request
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Store=request.form.get('Store'),
            Dept=request.form.get('Dept'),
            IsHoliday=request.form.get('IsHoliday'),
            Temperature =float(request.form.get('Temperature')),
            Fuel_Price =(request.form.get('Fuel_price')),
            CPI =float(request.form.get('CPI')),
            Unemployment = float(request.form.get('Unemployment')),
            Type=request.form.get('Type'),
            Size =float(request.form.get('Size')),
            year =request.form.get('year'),
            week=request.form.get('week')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)