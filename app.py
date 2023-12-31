from flask import Flask,render_template, request
import numpy as np
import pandas as pd 
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.components import data_transformation

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
            data_list=list(request.form.values())
        )
        pred_df=data.get_data_as_data_frame()
        print("app.py",pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)