import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the California Housing dataset from scikit-learn
from sklearn.datasets import fetch_california_housing

# Load and prepare the data
print("Loading California Housing dataset...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Linear Regression model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Root Mean Squared Error: ${rmse:.2f}k")
print(f"R² Score: {r2:.3f}")

# Feature importance (coefficients) plot
feature_importance = pd.DataFrame({
    'feature': housing.feature_names,
    'coefficient': model.coef_
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['coefficient'])
plt.title('Feature Coefficients in Linear Regression Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('linear_regression_coefficients.png')
print("\nFeature coefficients plot saved as 'linear_regression_coefficients.png'")
