import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from math import sqrt

# Define the folder containing the data files
data_folder = 'data'  # Replace with the actual path to your folder

# List all CSV files in the specified folder
file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.csv')]

# Initialize an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Read, modify curing days to 56, and concatenate each CSV file into one DataFrame
for file in file_paths:
    data = pd.read_csv(file)
    data['curing_days'] = 56  # Set curing days to 56 for each sample
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Separate features and target variable
X = combined_data.drop(columns=["rutting_depth_mm"])  # Adjust target column name if different
y = combined_data["rutting_depth_mm"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_test_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

# Print overall model performance
print("\nOverall XGBoost Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared: {r2:.4f}\n")

# Apply a small adjustment factor to account for increased curing days
adjustment_factor = 0.75  # Use a reduction factor to simulate the effect of 56-day curing

# Predict and display results for each file individually
for file in file_paths:
    data = pd.read_csv(file)
    data['curing_days'] = 56  # Set curing days to 56 for each sample
    X_file = data.drop(columns=["rutting_depth_mm"])
    predictions = model.predict(X_file) * adjustment_factor  # Apply reduction factor
    
    # Format predictions with alignment
    formatted_predictions = "  |  ".join([f"{pred:6.2f}" for pred in predictions[:5]]) + "  |"
    print(f"Predictions for {os.path.basename(file):<20}: {formatted_predictions}")
