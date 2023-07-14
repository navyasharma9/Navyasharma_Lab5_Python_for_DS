import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
from flask import *
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
Fuel_Type = {'0': 'Diesel', '1': 'Petrol', '2': 'CNG'}
Tansmission = {'0': 'Manual', '1': 'Automatic'}
Seller_Type = {'0': 'Dealer', '1': 'Individual', }

from flask import render_template
@app.route('/')
def home():
    return render_template("Index.html", title="Car Prediction", Fuel_Type=Fuel_Type, Tansmission=Tansmission,
                           Seller_Type=Seller_Type)


@app.route('/compute', methods=['Post'])
def compute():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaled_final_features = ss.fit_transform(final_features)
    prediction = model.predict(scaled_final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', message='Predicted value is {}'.format(output), title="Car Prediction",
                           Fuel_Type=Fuel_Type, Tansmission=Tansmission, Seller_Type=Seller_Type)


@app.route("/cookie")
def cookie():
    rsp = make_response("<h1>cookie created</h1>")
    rsp.set_cookie('working with', 'flask framework')
    return rsp


@app.route("/get-cookie")
def get_cookie():
    working_with = request.cookies.get('working with')
    return jsonify({'working_with': 'working with'})


if __name__ == "__main__":
    app.run(debug=True)