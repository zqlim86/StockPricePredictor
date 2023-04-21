import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import RootMeanSquaredError
import random
import tensorflow as tf
from flask import Flask, jsonify


def stockPrediction(ticker):

    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Fetch data from FMP API
    symbol = "AAPL" # Apple Inc.
    api_key = "853a3bc4cd9d6a7a2cf27666e3c70c0c"
    url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{ticker}?period=10&type=rsi&apikey={api_key}"
    response = requests.get(url)
    data = response.json()


    # Convert data to a Pandas dataframe and extract the close price and rsi
    df = pd.DataFrame(data)
    df = df.set_index('date')
    close_prices = df['close'].values.reshape(-1, 1)
    rsi_data = df['rsi'].values.reshape(-1, 1)
    close_prices = close_prices[::-1]
    rsi_data = rsi_data[::-1]


    print("Close Price: ")
    print(close_prices)
    print("\n\n")
    print("RSI: ")
    print(rsi_data)
    print("\n\n")

    # Combine close price and RSI data into a single input array
    input_data = np.concatenate((close_prices, rsi_data), axis=1)
    print(input_data)


    # Split the data into training and testing sets
    train_size = int(len(input_data) * 0.8)
    test_size = len(input_data) - train_size
    train_data, test_data = input_data[0:train_size,:], input_data[train_size:len(input_data),:]
    # print(train_data)

    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.fit_transform(test_data)

    test1 = scaler.inverse_transform(scaled_train_data)
    test2 = scaler.inverse_transform(scaled_test_data)

    print(scaled_train_data)
    print(test1)
    print(test2)


    # Define window size
    window_size = 30

    # Create input and output data
    def create_data(data, window_size):
        X, Y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, :])
            Y.append(data[i, :])
        return np.array(X), np.array(Y)


    train_X, train_Y = create_data(scaled_train_data, window_size)
    test_X, test_Y = create_data(scaled_test_data, window_size)

    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(window_size, 2)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False, input_shape=(window_size, 2)))
    model.add(Dropout(0.2))
    model.add(Dense(2))

    # Print the model summary
    model.summary()

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', RootMeanSquaredError()])


    # Train the model
    # history = model.fit(train_X, train_Y, epochs=100, batch_size=32, validation_split=0.2)
    model.fit(train_X, train_Y, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, mse, rmse = model.evaluate(test_X, test_Y, verbose=0)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)


    # Make predictions
    predictions = model.predict(test_X)


    #Denormalizing the predictions and test
    predictions_denormalize = scaler.inverse_transform(predictions)
    testY_denormalize = scaler.inverse_transform(test_Y)


    print(predictions_denormalize)
    print("\n\n")
    print(testY_denormalize)


    # Future predictions
    # Fetch the latest daily data and add it to the input data array
    url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{ticker}?period=10&type=rsi&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df = df.iloc[::-1]
    df = df.set_index('date')
    close_price = df['close'].values[-1].reshape(-1, 1)
    rsi = df['rsi'].values[-1].reshape(-1, 1)
    input_data = np.concatenate((close_price, rsi), axis=1)
    input_data = scaler.transform(input_data.reshape(1, -1))
    # print(df)
    # print("\n")

    # Use the latest daily data to predict the next day's price
    x = np.array([scaled_test_data[-window_size:, :]])
    pred = model.predict(x)
    pred_denormalize = scaler.inverse_transform(pred)
    print("X:")
    print(x)
    print('Predicted Close Price:', pred_denormalize[0][0])
    print("\n\n")

    print(predictions_denormalize)
    print("\n\n")

    # Get the latest date from the dataframe
    latest_date = pd.to_datetime(df.index[-1]).date()

    # Generate a sequence of 222 dates starting from the latest date
    dates = pd.date_range(start=latest_date, periods=222, freq='-1D')

    # Add one future date to the end of the sequence
    future_date = latest_date + pd.Timedelta(days=1)
    dates = dates.insert(0, future_date)

    # Convert the dates to string format in YY-MM-DD format
    dates = [date.strftime('%y-%m-%d') for date in dates]
    dates = dates[::-1]
    print(dates)

    # Concatenate the predicted value for the next day to the predictions array
    predictions_denormalize = np.concatenate((predictions_denormalize, pred_denormalize), axis=0)
    # print(predictions_denormalize)
    # print("\n\n")

    num_predictions = predictions_denormalize.shape[0]
    print("Number of Predictions:", num_predictions)

    # Concatenate the dates with the predictions array
    predictions_with_dates = np.column_stack((dates, predictions_denormalize))
    print(predictions_with_dates)
    return predictions_with_dates


app = Flask(__name__)


@app.route('/predict/<ticker>', methods=['POST'])
def get_predictions(ticker):

    # predictions = predictions_with_dates.tolist()
    predictions = stockPrediction(ticker).tolist()

    result = []
    for prediction in predictions:
        result.append({'date': prediction[0], 'close': prediction[1], 'rsi': prediction[2]})
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, port=5000)
    # #
    # # #
    # # # # Plot the predictions and actual values
    # # plt.plot(dates, predictions_denormalize[:, 0], label='Predicted Close Price')
    # # plt.plot(testY_denormalize[:, 0], label='Actual Close Price')
    # # # # plt.plot(dates[-len(predictions_denormalize):], predictions_denormalize[:, 0], label='Predicted Close Price')
    # # # # plt.plot(dates[-len(testY_denormalize):], testY_denormalize[:, 0], label='Actual Close Price')
    # # plt.legend()
    # # plt.show()
