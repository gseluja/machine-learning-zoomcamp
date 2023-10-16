import pickle

from flask import Flask
from flask import request
from flask import jsonify

# Load the model
model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as model_in:
    model = pickle.load(model_in)

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)


# Set up the method and endpoint of the application

app = Flask('app')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {"credit_probability": y_pred,
              }

    return jsonify(result)


if __name__=='__main__':
    app.run(host='0.0.0.0', port='9696', debug=True)
