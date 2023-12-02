import pandas as pd


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
