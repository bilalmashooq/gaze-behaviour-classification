# train_and_save_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os


def load_data(gaze_features_path, blink_results_path):
    """Load and merge gaze_features and blink_results datasets."""
    gaze_features = pd.read_csv(gaze_features_path)
    blink_results = pd.read_csv(blink_results_path)
    merged_data = pd.merge(gaze_features, blink_results, on=['person_id', 'label'], how='inner')
    return merged_data


def encode_labels(merged_data):
    """Encode the target variable 'label' into numerical values."""
    le = LabelEncoder()
    merged_data['label_encoded'] = le.fit_transform(merged_data['label'])  # 0: scripted, 1: spontaneous
    return merged_data, le


def define_features(merged_data):
    """Define feature matrix X and target vector y."""
    X = merged_data.drop(['person_id', 'label', 'label_encoded'], axis=1)
    y = merged_data['label_encoded']
    return X, y


def create_pipeline():
    """Create a machine learning pipeline with feature selection, scaling, and classification."""
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42))
    ])
    return pipeline


def evaluate_model(pipeline, X, y, skf):
    """Evaluate the model using cross-validation and display performance metrics."""
    # Perform cross-validation and get predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Scripted', 'Spontaneous']))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Scripted', 'Spontaneous'],
                yticklabels=['Scripted', 'Spontaneous'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.show()

    # ROC Curve and AUC
    y_scores = cross_val_predict(pipeline, X, y, cv=skf, method='decision_function')
    auc = roc_auc_score(y, y_scores)
    fpr, tpr, thresholds = roc_curve(y, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc='lower right')
    plt.show()


def train_and_save(pipeline, X, y, skf, model_path, label_encoder_path):
    """Train the model on the entire dataset and save the pipeline and label encoder."""
    # Fit the pipeline to the entire dataset
    pipeline.fit(X, y)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)

    # Save the pipeline
    joblib.dump(pipeline, model_path)
    print(f"Model pipeline saved to {model_path}")

    # Save the label encoder
    joblib.dump(le, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")


if __name__ == "__main__":
    # Define file paths
    gaze_features_path = r"C:\Users\muham\PycharmProjects\gazelle\gaze_output\gaze_features.csv"
    blink_results_path = r"C:\Users\muham\PycharmProjects\gazelle\scripts\blink_results.csv"

    # Load and merge data
    merged_data = load_data(gaze_features_path, blink_results_path)

    # Encode labels
    merged_data, le = encode_labels(merged_data)

    # Define features and target
    X, y = define_features(merged_data)

    # Create pipeline
    pipeline = create_pipeline()

    # Define cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate model
    evaluate_model(pipeline, X, y, skf)

    # Define paths to save the model and label encoder
    model_path = r"C:\Users\muham\PycharmProjects\gazelle\models\behavior_classifier_pipeline.joblib"
    label_encoder_path = r"C:\Users\muham\PycharmProjects\gazelle\models\label_encoder.joblib"

    # Train on the entire dataset and save the model
    train_and_save(pipeline, X, y, skf, model_path, label_encoder_path)
