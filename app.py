# app.py

from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    np_array = [x for x in request.form.values()]
    # print(np_array)
    final_features = ",".join(np_array)
    # final_features = [np.array(int_features)]
    np_array = np.asarray(final_features.split(','), dtype=np.float32)
    ####################################
    reshaped_array = np_array.reshape(1,-1)
  # print(reshaped_array.shape)
    
    # sd_scalar = StandardScaler()

    # Std_data = sd_scalar.transform(reshaped_array)
    
    ####################################

    
    # Make prediction
    prediction = model.predict(reshaped_array)
    print(prediction[0])
    output = 'Diabetes' if prediction[0] == 1 else 'Non-Diabetes'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
