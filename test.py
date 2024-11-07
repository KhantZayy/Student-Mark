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
# for i in range(len(y_predict)):
#     print(f"True Value: {testY.iloc[i]}, Features: {testX.iloc[i].values}, Predicted Value: {y_predict[i]}")
number_couses = float(input('Enter number of courses : '))
study_time = float(input('Enter study time : '))
data = model.predict(pd.DataFrame({'number_courses': [number_couses],
                        'time_study': study_time}))
print(f'Predicted Mark: {data}')