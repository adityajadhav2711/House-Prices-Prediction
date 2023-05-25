from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
df = pd.read_csv('cleaned_data.csv')
model = pickle.load(open('clf.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    total_sqft = request.form.get('total_sqft')
    bath = request.form.get('bath')
    site_location = request.form.get('site_location')
    bhk = request.form.get('bhk')
    
    
    print(total_sqft,bath,site_location,bhk)
    
    input = pd.DataFrame([[total_sqft,bath,site_location,bhk]],columns=['total_sqft','bath','site_location','bhk'])

    prediction = model.predict(input)[0]

    return 'Price of the house is ' + str(round(prediction*100000,2)) 

if __name__ == '__main__':
    app.run(debug=True)