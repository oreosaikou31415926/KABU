import subprocess
import sys
subprocess.run(["pip","install","streamlit","-t","/home/appuser"])
subprocess.run(["pip","install","yfinance","-t","/home/appuser"])
subprocess.run(["pip","install","pandas","-t","/home/appuser"])
subprocess.run(["pip","install","numpy","-t","/home/appuser"])
subprocess.run(["pip","install","scikit-learn","-t","/home/appuser"])
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

Input_1 = []
Input0 = []
Input1 = []
Input3 = []
Input4 = []
Input2 = []
A = 2
Output = []


def get(ticker):
  data = yf.download(ticker, period='max', interval="1d")
  print(data)
  for i in range(len(data["Open"]) - A):
    Input1.append(int(data["Open"].iloc[i] * 100) / 100)
  for i in range(len(data["High"]) - A):
    Input2.append(int(data["High"].iloc[i] * 100) / 100)
  for i in range(len(data["Low"]) - A):
    Input3.append(int(data["Low"].iloc[i] * 100) / 100)
  for i in range(len(data["Open"]) - A):
    Input4.append(int(data["Close"].iloc[i] * 100) / 100)
  for i in range(len(data["High"]) - A):
    Output.append(((int(data["High"].iloc[i + 2] * 100) / 100)) / ((int(data["High"].iloc[i + 1] * 100) / 100)))


def gakusyuu():
  X = dict(A1=Input1, A2=Input2, A3=Input3, A4=Input4)
  X = pd.DataFrame(data=X)
  y = dict(Output=Output)
  y = pd.DataFrame(data=y)
  y = np.ravel(y)
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=777)
  clf = LinearRegression()
  clf.fit(X_train, y_train)
  pred = clf.predict(X_test)
  return clf


def yosoku(ticker, clf):
  data = yf.download(ticker, period='max', interval="1d")
  Input1 = [int(data["Open"].iloc[len(data["Open"]) - 1] * 100) / 100]
  Input2 = [int(data["High"].iloc[len(data["High"]) - 1] * 100) / 100]
  Input3 = [int(data["Low"].iloc[len(data["Low"]) - 1] * 100) / 100]
  Input4 = [int(data["Close"].iloc[len(data["Close"]) - 1] * 100) / 100]
  print(Input1)
  print(Input2)
  print(Input3)
  X_test = dict(A1=Input1, A2=Input2, A3=Input3, A4=Input4)
  X_test = pd.DataFrame(data=X_test)
  Output = clf.predict(X_test)
  if Output > 1:
    print(str(Output*100) + "%上昇する見込みあり。")
  else:
    print(str(Output*100) + "%下がる見込みあり。")
  return ("明日のHighは今日の"+str(Output*100) + "%になると予測されます。")
get("GOOGL")
st.text_input("Message", key="text_input")
def change_value():
  meigara = st.session_state["text_input"]
  get(meigara)
  clf = gakusyuu()
  st.session_state["text_input"] = yosoku(meigara,clf)
st.button("Click",on_click=change_value)
