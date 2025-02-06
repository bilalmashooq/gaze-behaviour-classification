# predict_behavior.py

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Suppress any potential warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')


def load_model_and_encoder(model_path, label_encoder_path):
    """
    Load the trained pipeline and label encoder from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")

    pipeline = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    print(f"Loaded model pipeline from {model_path}")
    print(f"Loaded label encoder from {label_encoder_path}")
    return pipeline, le


def load_new_data(new_data_path):
    """
    Load new data for prediction.
    """
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"New data file not found at {new_data_path}")

    new_data = pd.read_csv(new_data_path)
    print(f"Loaded new data from {new_data_path} with shape {new_data.shape}")
    return new_data


def preprocess_new_data(new_data):
    """
    Preprocess new data to match training features.
    """
    # Drop any non-feature columns if present (e.g., 'person_id')
    # Assuming new_data has the same feature columns as training data (excluding 'person_id', 'label', etc.)
    # Adjust based on your actual new_data structure
    columns_to_drop = ['person_id', 'label']  # Add or remove columns as necessary
    for col in columns_to_drop:
        if col in new_data.columns:
            new_data = new_data.drop(columns=col)

    # Handle missing values if any (e.g., impute with mean)
    if new_data.isnull().sum().any():
        new_data = new_data.fillna(new_data.mean())
        print("Missing values found and imputed with column means.")

    return new_data


def make_predictions(pipeline, new_data, label_encoder):
    """
    Make predictions on new data using the trained pipeline.
    """
    predictions_numeric = pipeline.predict(new_data)
    predictions_proba = pipeline.predict_proba(new_data)[:, 1]  # Probability for class '1' (spontaneous)

    # Decode numerical predictions to original labels
    predictions_labels = label_encoder.inverse_transform(predictions_numeric)

    # Combine predictions with the new data
    prediction_results = new_data.copy()
    prediction_results['predicted_label'] = predictions_labels
    prediction_results['confidence_score'] = predictions_proba

    return prediction_results


def save_predictions(prediction_results, output_path):
    """
    Save the prediction results to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prediction_results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    # Define paths
    model_path = r"C:\Users\muham\PycharmProjects\gazelle\models\behavior_classifier_pipeline.joblib"
    label_encoder_path = r"C:\Users\muham\PycharmProjects\gazelle\models\label_encoder.joblib"
    new_data_path = r"C:\Users\muham\PycharmProjects\gazelle\data\new_data.csv"  # Replace with your new data path
    output_predictions_path = r"C:\Users\muham\PycharmProjects\gazelle\predictions\new_data_predictions.csv"

    # Load the model and label encoder
    pipeline, le = load_model_and_encoder(model_path, label_encoder_path)

    # Load new data
    new_data = load_new_data(new_data_path)

    # Preprocess new data
    preprocessed_data = preprocess_new_data(new_data)

    # Ensure that the new data has the same number of features as the training data
    if preprocessed_data.shape[1] != pipeline.named_steps['feature_selection'].get_support().sum():
        raise ValueError(
            f"New data has {preprocessed_data.shape[1]} features, but the model expects {pipeline.named_steps['feature_selection'].get_support().sum()} features.")

    # Make predictions
    prediction_results = make_predictions(pipeline, preprocessed_data, le)

    # Save predictions
    save_predictions(prediction_results, output_predictions_path)

    # Optionally, display the first few predictions
    print("\nSample Predictions:")
    print(prediction_results.head())


if __name__ == "__main__":
    main()
