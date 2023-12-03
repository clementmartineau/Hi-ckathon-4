import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np

from tools import load_data

dataframes = load_data.load_data()



# Assuming the target variable is "Month 4"
target_col = "Month 4"

# Extract features and target variable
X = dataframes["train-data"].drop(columns=[target_col])
y = dataframes["train-data"][target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
model = xgb.XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Now you can use the trained model to make predictions on new data
# For example, if you have new data in a DataFrame called 'new_data'
# new_predictions = model.predict(new_data)

# If you want to predict for the entire dataset, you can use the following
# predictions = model.predict(X)

# Optionally, you can also visualize feature importances
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f in range(X.shape[1]):
    print(f"{feature_names[indices[f]]}: {importances[indices[f]]}")
