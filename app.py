from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

#Create flask app
app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))

#Home page
@app.route ('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

        # Get data from the request
        data = [x for x in request.form.values()]

        # Convert data to numpy array
        input_data = [np.array(data)]

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction as html page
        return render_template('output.html',prediction_text="The premium is".format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
