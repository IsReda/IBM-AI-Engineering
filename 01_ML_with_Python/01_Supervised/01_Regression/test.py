import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)
print(df.head())
print(df.isna().sum())

y = df["CO2EMISSIONS"].values.astype('float32')
X = df.drop("CO2EMISSIONS", axis=1).values

X = df.select_dtypes(include=['number'])
X = normalize(X, axis=1, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeRegressor(criterion= 'squared_error', max_depth=4, random_state=35)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

print("R2 Score: ", R2)
print("Mean Squared Error: ", MSE)

