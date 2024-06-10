import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, request, jsonify

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
app = Flask(_name_)

# Load the model
model = joblib.load(catboost_model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    app.run(debug=True)