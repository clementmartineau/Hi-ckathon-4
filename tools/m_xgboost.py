from catboost import CatBoostRegressor

 
import pandas as pd
import numpy as np

file_paths = [
    "X_train", "X_train", "y_train", "y_train"
]
data  = []

for file_path in file_paths:
    data += [pd.read_csv(file_path + ".csv")]
    data[-1] = data[-1].drop(columns=["Unnamed: 0"])

data[0]["Month 1"] = data[0]["Month 1"].apply(lambda x : int(x.replace(" ", "")))
data[0]["Month 2"] = data[0]["Month 2"].apply(lambda x : int(x.replace(" ", "")if type(x)!=int else x))
data[0]["Month 3"] = data[0]["Month 3"].apply(lambda x : int(x.replace(" ", "")))
data[1]["Month 1"] = data[1]["Month 1"].apply(lambda x : int(x.replace(" ", "")))
data[1]["Month 2"] = data[0]["Month 2"].apply(lambda x : int(x.replace(" ", "")if type(x)!=int else x))
data[1]["Month 3"] = data[1]["Month 3"].apply(lambda x : int(x.replace(" ", "")))


data[2]['Month 4']  = data[2]['Month 4'] .apply(lambda x : int(x.replace(" ", "")))
data[3]['Month 4']  = data[3]['Month 4'] .apply(lambda x : int(x.replace(" ", "")))

### !!! Must do the same for Month 4 in Y, refactor and put in load_data !!!
###
###
###
##
#

X_train = data[0]
X_val = data[1]
y_train = data[2]
y_val = data[3]

# Create the XGBoost regression model
xgb_model = CatBoostRegressor(iterations=1000,
                           task_type="GPU")

# Train the model
xgb_model.fit(X_train, y_train, verbose=True)
X_train.info()
xgb_model