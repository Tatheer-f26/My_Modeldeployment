import pickle
import numpy as np

from flask import Flask, request, jsonify
from pycaret.anomaly import *
model = pickle.load(open('modelf.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])

def predict():
    #Roomtemp = request.form.get('Roomtemp')
    HeartRate = request.form.get('HeartRate')
    #Bodytemp = request.form.get('Bodytemp')
    GSR = request.form.get('GSR')

    input_query = np.array([[HeartRate, GSR]],dtype=float)

    result = model.predict(input_query)[0]

    return jsonify({'meltdown': str(result)})

if __name__ == '__main__':
    app.run(debug=True)