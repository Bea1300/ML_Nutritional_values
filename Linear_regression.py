import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
dataset_path = 'dataset/food.csv' 
df = pd.read_csv(dataset_path)

print(df.head())

X = df.drop(['Data.Kilocalories', 'Description', 'Nutrient Data Bank Number'], axis=1)
y = df['Data.Kilocalories']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

# Plotting actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Line of Best Fit')
plt.title("Actual vs Predicted Kilocalories")
plt.show()

