import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and prepare the data
data = pd.read_csv('advertising.csv')
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate coefficients
coefficients = pd.DataFrame({'Medium': X.columns, 'Impact': model.coef_})
coefficients = coefficients.sort_values('Impact', ascending=False)

# Visualizations
plt.figure(figsize=(12, 5))

# Actual vs Predicted Sales
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')

# Feature Importance
plt.subplot(1, 2, 2)
plt.bar(coefficients['Medium'], coefficients['Impact'])
plt.title('Advertising Medium Impact on Sales')
plt.xlabel('Medium')
plt.ylabel('Impact')

plt.tight_layout()
plt.show()

# Print results
print("\n=== Advertising Impact on Sales Analysis ===\n")

print("Model Performance:")
print(f"R-squared Score: {r2:.2%}")
print(f"Mean Squared Error: {mse:.2f}")

print("\nImpact of Advertising Mediums on Sales:")
for index, row in coefficients.iterrows():
    print(f"{row['Medium']}: {row['Impact']:.4f}")

print("\nInterpretation:")
for index, row in coefficients.iterrows():
    print(f"- A $1,000 increase in {row['Medium']} advertising is associated with a ${row['Impact']*1000:.2f} increase in sales.")

print("\nRecommendations:")
top_medium = coefficients.iloc[0]['Medium']
second_medium = coefficients.iloc[1]['Medium']
least_medium = coefficients.iloc[2]['Medium']

print(f"1. Prioritize {top_medium} advertising for the highest impact on sales.")
print(f"2. Allocate a significant portion of the budget to {second_medium} advertising as well.")
if coefficients.iloc[2]['Impact'] < coefficients.iloc[1]['Impact'] / 2:
    print(f"3. Consider reducing or reallocating budget from {least_medium} advertising, as it has significantly less impact.")
else:
    print(f"3. Maintain a balanced approach with {least_medium} advertising, but prioritize the top two mediums.")

print("\nNext Steps:")
print("1. Conduct A/B testing to validate these findings in real-world scenarios.")
print("2. Analyze the cost-effectiveness of each advertising medium.")
print("3. Regularly update this model with new data to ensure ongoing accuracy.")