import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
clf = pickle.load(open('clf.pkl', 'rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Pune House Price prediction")

# total_sqft
total_sqft = st.number_input('Total sqft area')

# bath
bath = st.number_input('Total no. of bathrooms')

# Location
site_location = st.selectbox('Location', df['site_location'].unique())

# bhk
bhk = st.number_input('Total no. of bedrooms')


if st.button('Predict Price'):

    query = np.array([total_sqft,bath,site_location,bhk])

    query = query.reshape(1,4)

    query= pd.DataFrame(data=query, index=np.arange(len(query)),columns=['total_sqft','bath','site_location','bhk'])

    st.title('Price of the House is '+ str(int((clf.predict(query)[0]*100000))))

    