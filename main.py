import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras._tf_keras.keras.models import load_model
import joblib

# models available
linear_model = "Saved_Linear_Model.pkl"
random_model = "Saved_Random_Model.pkl"
lstm_model = "Saved_LSTM_Model.h5"
gru_model = "Saved_GRU_Model.h5"

# get dataset
data_MCD = pd.read_csv("Preprocessed_MCD.csv")
d_train_reg = pd.read_csv("MCD_train_data.csv")
d_test_reg = pd.read_csv("MCD_test_data.csv")

# get features
x = data_MCD[["High", "Low", "Open", "Volume"]]
x_train_reg = d_train_reg[["High", "Low", "Open", "Volume"]]
x_test_reg = d_test_reg[["High", "Low", "Open", "Volume"]]
y = data_MCD["Close"]

# default scaling setting for data
scale_X = MinMaxScaler()
scale_X_reg = MinMaxScaler()
scale_y = MinMaxScaler()
normalised_X =scale_X.fit_transform(x)
norm_train_reg = scale_X_reg.fit_transform(x_train_reg)
norm_test_reg = scale_X_reg.transform(x_test_reg)
normalised_y =scale_y.fit_transform(y.values.reshape(-1,1))

# window 
window_size = 100

# MCD stock price
def show_close(data):
    data["Date"] = pd.to_datetime(data["Date"])
    plt.figure(figsize=(14,6))
    plt.plot(data["Date"][int(0.9*len(data)):], data["Close"][int(0.9*len(data)):])
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.title("MCD Stock Price (Close)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig("MCD Stock Price (Close)")
    plt.show()

# windowing for RNN
def window_for_RNN(X, y, window):
    start_test = int(0.9*len(X))-100
    X_test = []
    y_test = []
    for i in range(window, len(X)):
        X_test.append(X[i-window:i, 0:4])
        y_test.append(y[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test[start_test:]
    y_test = y_test[start_test:]
    return X_test, y_test, start_test

# Plot graph (Actual vs Predicted)
def plotting_graph(pva, y_test, prediction, label):
    plt.figure(figsize=(14,6))
    plt.plot(pva.index, y_test, label="Actual Price", color="red")
    plt.plot(pva.index, prediction, label="Predicted Price", color="blue")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(label)
    plt.legend()
    plt.savefig(f"{label}.png")
    plt.show()

# Calculating performance 
def performance(y, y_predict, label):
    mse = mean_squared_error(y, y_predict)
    mae = mean_absolute_error(y, y_predict)
    r_square = r2_score(y, y_predict)
    print(f"\n{label} prediction statistics:")
    print(f"Mean Square Error = {mse}")
    print(f"Mean Absolute Error = {mae}")
    print(f"R^2 value = {r_square}\n\n")

# Linear Regression Prediction
def prediction_linear(model, X_test, y_test, data):
    y_test = y_test[int(0.9*len(data)):]
    print("\nBEGIN LINEAR REGRESSION PREDICTION.....")
    linear = joblib.load(model)
    predict_linear = linear.predict(X_test)

    # create dataframe for actual vs prediction
    pva = pd.DataFrame(y_test.values, columns=["Actual"], index=data["Date"][int(0.9*len(data)):])
    pva["Predicted"] = predict_linear
    pva.index = pd.to_datetime(pva.index)
    pva.index.name = "Date"
    print(pva)

    # plot graph for actual vs prediction
    label = "Prediction (Linear Regression)"
    plotting_graph(pva, y_test, predict_linear, label)

    # performance
    performance(y_test, predict_linear, label="Linear Regression")



# Random Forest prediction
def prediction_random(model, X_test, y_test, data):
    y_test = y_test[int(0.9*len(data)):]
    print("BEGIN RANDOM FOREST PREDICTION.....")
    random = joblib.load(model)
    predict_random = random.predict(X_test)

    # create dataframe for actual vs prediction
    pva = pd.DataFrame(y_test.values, columns=["Actual"], index=data["Date"][int(0.9*len(data)):])
    pva["Predicted"] = predict_random
    pva.index = pd.to_datetime(pva.index)
    pva.index.name = "Date"
    print(pva)

    # plot graph for actual vs prediction
    label = "Prediction (Random Forest)"
    plotting_graph(pva, y_test, predict_random, label)

    # performance
    performance(y_test, predict_random, label="Random Forest")



# LSTM prediction
def prediction_lstm(model, X_test, y_test, scale_y, data, start):
    print("BEGIN LSTM PREDICTION.....")
    LSTM_model = load_model(model)
    predict_lstm = LSTM_model.predict(X_test)
    predict_lstm = scale_y.inverse_transform(predict_lstm)
    y_test = scale_y.inverse_transform(y_test.reshape(-1,1))

    # create dataframe for actual vs predicted
    date_for_test = data["Date"].iloc[start+100:]
    pva = pd.DataFrame(y_test, columns=["Actual"], index=date_for_test)
    pva["Predicted"] = predict_lstm
    pva.index = pd.to_datetime(pva.index)
    pva.index.name = "Date"
    print(pva)

    # plot graph for actual vs prediction
    label = "Prediction (LSTM)"
    plotting_graph(pva, y_test, predict_lstm, label)

    # performance
    performance(y_test, predict_lstm, label="LSTM")
    


# GRU prediction
def prediction_gru(model, X_test, y_test, scale_y, data, start):
    print("BEGIN GRU PREDICTION.....")
    GRU_model = load_model(model)
    predict_gru = GRU_model.predict(X_test)
    predict_gru = scale_y.inverse_transform(predict_gru)
    y_test = scale_y.inverse_transform(y_test.reshape(-1,1))

    # create dataframe for actual vs predicted
    date_for_test = data["Date"].iloc[start+100:]
    pva = pd.DataFrame(y_test, columns=["Actual"], index=date_for_test)
    pva["Predicted"] = predict_gru
    pva.index = pd.to_datetime(pva.index)
    pva.index.name = "Date"
    print(pva)

    # plot graph for actual vs prediction
    label = "Prediction (GRU)"
    plotting_graph(pva, y_test, predict_gru, label)

    # performance
    performance(y_test, predict_gru, label="GRU")



# Function calling
def main():
    show_close(data_MCD)
    prediction_linear(linear_model, norm_test_reg, y, data_MCD)
    prediction_random(random_model, norm_test_reg, y, data_MCD)
    X_test, y_test, start = window_for_RNN(normalised_X, normalised_y, window_size)
    prediction_lstm(lstm_model, X_test, y_test, scale_y, data_MCD, start)
    prediction_gru(gru_model, X_test, y_test, scale_y, data_MCD, start)

main()