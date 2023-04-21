# from flask import Flask, jsonify, request
# from main import stockPrediction
# # from main import predictions_with_dates
#
# app = Flask(__name__)
#
#
# @app.route('/predict/<ticker>', methods=['POST'])
# def get_predictions(ticker):
#
#     # predictions = predictions_with_dates.tolist()
#     predictions = stockPrediction(ticker).tolist()
#
#     result = []
#     for prediction in predictions:
#         result.append({'date': prediction[0], 'close': prediction[1], 'rsi': prediction[2]})
#     return jsonify(result)
#
#
# if __name__ == '__main__':
#     app.run(debug=False, port=5000)
