import sklearn
import numpy
import pandas
import joblib
import flask
import xgboost
from flask import Flask , request
pipeline= joblib.load("model_final.pkl")
print(pipeline)


app= Flask('__name__')
@app.route('/predict', methods=['POST'])
def predict():
  
  df = pandas.DataFrame(request.json)
  resultat= pipeline.predict(df)[0]
  return(str(resultat),201)

 
@app.route('/ping', methods=['GET'])
def ping():
    return ('pong', 200)



@app.route('/')
def index():
    return "<h1>bienvenue dans notre api </h1>"

if __name__ == "__main__":
  app.run(host='0.0.0.0')