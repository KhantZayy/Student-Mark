import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

data = pd.read_csv('Student_Marks_new.csv')
data.dropna(axis=0 ,inplace= True)

x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

trainX, testX, trainY, testY = train_test_split(x, y, test_size= 0.1)

model = LinearRegression()
model.fit(trainX, trainY)

y_predict = model.predict(testX)

st.title(':blue[Login Session]')
st.text('Enter your Student [ID, Username and Password].')

username = 'khant'
password = 123

username_input = st.text_input('Username : ')
password_input = st.number_input('Password : ')

if username == username_input and password == password_input:
    st.title('Mark Prediction')
    number_course = st.number_input('Enter number of courses : ')
    time_study = st.number_input('Enter your study time : ')
    predict = model.predict(pd.DataFrame({'number_courses' : [number_course],
                            'time_study' : time_study}))
    st.write(f'Predicted Mark {predict}')
else:
    st.write('Wrong username or password')
