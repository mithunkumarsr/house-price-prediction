import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(
    open('/Users/mithunkumar/Desktop/HomePrices/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The house price predicted is $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
