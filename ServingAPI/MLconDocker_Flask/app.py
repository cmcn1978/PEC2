from flask import Flask, request, jsonify, render_template, session, redirect, url_for, session
import requests
import pandas as pd
import numpy as np
from my_model.predict import make_prediction


app = Flask(__name__, template_folder='template')
@app.route('/',  methods = ['GET','POST'])
def home():
    print('request.method:', request.method)
    if request.method == 'POST':  
        # Get values through input bars
        pclass = request.form.get("pclass")
        age = request.form.get("age")
        name = request.form.get("name")
        sex = request.form.get("sex")
        sibsp = request.form.get("sibsp")
        parch = request.form.get("parch")
        ticket = request.form.get("ticket")
        fare = request.form.get("fare")
        cabin = request.form.get("cabin")
        embarked = request.form.get("embarked")

        # Put inputs to dataframe
        X = pd.DataFrame([[pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked]], 
                            columns =  ["pclass",
                                        "name",
                                        "sex",
                                        "age",
                                        "sibsp",
                                        "parch",
                                        "ticket",
                                        "fare",
                                        "cabin",
                                        "embarked"])
        print('X:', X)
        # Get prediction
        prediction = make_prediction(X)
    else:
        prediction = ""
    if prediction == 1:
        prediction = "Survived"
    else:
        prediction = "Non-survived"
    return render_template("website.html", output = prediction)


if __name__ =='__main__':
    app.run(debug=True,host="0.0.0.0", port=int("5000"))