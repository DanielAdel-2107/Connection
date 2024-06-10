import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, request, jsonify
import os

# Load data and preprocess (as per your original script)
df = pd.read_csv('kc_house_data.csv')
df.drop(columns=['id', 'date', 'sqft_basement', 'yr_renovated'], inplace=True)
y = df['price']
x = df.drop(columns=['price'])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Train CatBoost model
catb_reg = CatBoostRegressor(loss_function='RMSE', iterations=1000, depth=3, colsample_bylevel=.4,
                             eval_metric='MAE', subsample=.5, random_state=0, verbose=0, eta=.1)
catb_reg.fit(x_train, y_train)

# Save the model using joblib
catboost_model_file = 'catboost_model.pkl'
joblib.dump(catb_reg, catboost_model_file)

# Create Flask application
app = Flask(__name__)

# Load the model
model = joblib.load(catboost_model_file)

@app.route('/')
def home():
    return "Welcome to the House Price Prediction API. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        # Extract features from JSON and convert to numpy array
        features = np.array(data['features']).reshape(1, -1)
        # Predict using the loaded model
        prediction = model.predict(features)
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # Handle any errors that occur
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
