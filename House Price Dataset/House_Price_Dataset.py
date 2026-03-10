#House Price Dataset
import pandas as pd  # used handling datasets
import numpy as np   # used numerical operations
import matplotlib.pyplot as plt  # used plotting graphs

from sklearn.model_selection import train_test_split  # used split data
from sklearn.linear_model import LinearRegression    # used linear regression model
from sklearn.metrics import mean_squared_error, r2_score  # used evaluate model

data = pd.read_csv("house_data.csv")

print("Here are the first 5 rows of the dataset:")
print(data.head())

X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Intercept (starting value):", model.intercept_)
print("Model Coefficients (impact of each feature):", model.coef_)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error (how wrong predictions are on average):", mse)
print("R2 Score (how well model explains data):", r2)

sqft = float(input("\nEnter the house square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

new_house = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=['sqft', 'bedrooms', 'bathrooms'])

predicted_price = model.predict(new_house)

print("\nPredicted House Price is:", predicted_price[0])

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()