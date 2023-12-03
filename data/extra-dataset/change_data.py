import pandas as pd


file = "data/extra-dataset/epi2022results05302022.csv"
dataframe = pd.read_csv(file, sep=",", header="infer")
columns_to_keep = ['code', 'iso', 'country', 'COE.new', 'COE.old', 'COE.change', 'COE.rnk.new', 'COE.rnk.old']
dataframe = dataframe[columns_to_keep]
print(dataframe)