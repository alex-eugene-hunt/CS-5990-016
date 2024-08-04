# togt_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import glob

# Load the data from multiple CSV files
def load_data(file_pattern):
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    print(f"Found files: {all_files}")
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# Train a supervised learning model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.9, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse:.2f}")
    return model

# Save the trained model to a file
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Load the trained model from a file
def load_model(file_path):
    return joblib.load(file_path)

# Predict and save results to a CSV file
def predict_and_save(model, file_pattern, output_csv, expected_features):
    new_data = load_data(file_pattern)
    # Ensure the new data contains only the expected features
    new_data = new_data[expected_features]
    predictions = model.predict(new_data)
    predictions_df = pd.DataFrame(predictions, columns=['p_x', 'p_y', 'p_z'])
    result_df = pd.concat([new_data, predictions_df], axis=1)
    result_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Training the model
    train_file_pattern = '../../resources/trajectory/*.csv'  # Replace with the actual path pattern to your training CSV files
    combined_data = load_data(train_file_pattern)
    
    # Assuming 'p_x', 'p_y', 'p_z' are the labels
    label_columns = ['p_x', 'p_y', 'p_z']
    # Assuming the state features include control inputs and other relevant columns
    feature_columns = [col for col in combined_data.columns if col not in label_columns]
    
    features = combined_data[feature_columns]
    labels = combined_data[label_columns]
    
    model_file = 'gate_traversal_model.joblib'
    
    model = train_model(features, labels)
    save_model(model, model_file)

    # Predicting and saving results
    predict_file_pattern = '../../resources/predict/predict_me.csv'  # Replace with the path pattern to your new input data CSV files
    output_csv = '../../resources/predict/predict_here.csv'  # The path where the predictions will be saved
    
    model = load_model(model_file)
    predict_and_save(model, predict_file_pattern, output_csv, feature_columns)
