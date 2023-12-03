import pandas as pd
import numpy as np

def split_date(date_str):
    if pd.isna(date_str):
        return None, None
    try:
        # Splitting the string to get the month and year
        splitted = date_str.split()
        #print("splitted : ",splitted)
        month_range, year = splitted



        # Returning the quarter and year in a standard format
        return year, month_range
    except Exception as e:
        # In case of any parsing error, return None
        return None, None
    
def load_data():
    file_paths = [
        "data/train-data.csv",
        "data/X_test.csv",
        "data/y_test_example.csv",
        "data/extra-dataset/worldbank_economic_data.csv",
        "data/extra-dataset/GSCPI_data.csv",
        "data/extra-dataset/LPIextend.csv",
        "data/extra-dataset/worldbank_economic_data.csv",
        "data/extra-dataset/worldbank_inflation_data.csv",
    ]

    # Create a dictionary to store DataFrames
    dataframes = {}

    # Specify parameters for reading CSV files with headers and different separators
    csv_params = {
        "data/train-data.csv": {"sep": ";", "header": "infer"},
        "data/X_test.csv": {"sep": ";", "header": "infer"},
        "data/y_test_example.csv": {"sep": ",", "header": "infer"},
        "data/extra-dataset/GSCPI_data.csv": {"sep": ",", "header": "infer"},
        "data/extra-dataset/LPIextend.csv": {"sep": ",", "header": "infer"},
        "data/extra-dataset/worldbank_economic_data.csv": {"sep": ",", "header": "infer"},
        "data/extra-dataset/worldbank_inflation_data.csv": {"sep": ",", "header": "infer"},
    }

    # Load each CSV file into a DataFrame with specified parameters
    for file_path in file_paths:
        key = file_path.split("/")[-1].split(".")[0]
        dataframes[key] = pd.read_csv(file_path, **csv_params[file_path])

    # replace NaN values with NL (=Normal Life)
    dataframes["train-data"]["Product Life cycel status"].fillna(value="NL", inplace=True)

<<<<<<< HEAD
# Load each CSV file into a DataFrame with specified parameters
for file_path in file_paths:
    key = file_path.split("/")[-1].split(".")[0]
    dataframes[key] = pd.read_csv(file_path, **csv_params[file_path])
dataframe = dataframes["train-data"]
# replace NaN values with NL (=Normal Life)
dataframe["Product Life cycel status"].fillna(value="NL", inplace=True)

# need to drop the row where q2 = 1, 2023, 
# because we don't have value for the 4th month, so it is not useful in our training
dataframe.dropna(inplace=True)

#get the month range and year from Date column
dataframe["year"], dataframe["month_range"] = zip(*dataframe["Date"].apply(split_date))

# drop the date column
dataframe.drop(columns=["Date"], inplace=True)

# Frequency embedding for Reference proxy
reference_value_counts= dataframe['Reference proxy'].value_counts()
reference_proxy_freq = reference_value_counts.apply(lambda x : np.log(x)/np.log(max(reference_value_counts)))
product_frequency = dataframe['Reference proxy'].map(reference_proxy_freq)

# Applying one-hot encoding to the selected columns
# Categorical data
columns_to_drop = ['Reference proxy', 'id_product', 'index', 'Cluster']
dataframe = dataframe.drop(columns=columns_to_drop)
columns_to_encode = dataframe.columns.difference(['Month 1', 'Month 2', 'Month 3', 'Month 4'])

dataframe = pd.get_dummies(dataframe, columns=columns_to_encode)
dataframe["product_frequency"] = product_frequency

from sklearn.model_selection import train_test_split

# Define the features and target variable
X = dataframe.drop(['Month 4'], axis=1)  # Features (excluding 'Month 4')
y = dataframe['Month 4']                 # Target variable ('Month 4')

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('X_train.csv')
X_val.to_csv('X_val.csv')
y_train.to_csv('y_train.csv')
y_val.to_csv('y_val.csv')

=======
    # need to drop the row where q2 = 1, 2023, 
    # because we don't have value for the 4th month, so it is not useful in our training
    dataframes["train-data"].dropna(inplace=True)

    #get the month range and year from Date column
    dataframes["train-data"]["year"], dataframes["train-data"]["month_range"] = zip(*dataframes["train-data"]["Date"].apply(split_date))

    # drop the date column
    dataframes["train-data"].drop(columns=["Date"], inplace=True)

    # one hot encoding for month range
    dataframes["train-data"] = pd.get_dummies(dataframes["train-data"], columns = ['month_range']) 
    print(dataframes["train-data"].isna().sum())

    return dataframes
>>>>>>> 1df6f2028d36592d2c350e3b8ff518f60e14af4a
