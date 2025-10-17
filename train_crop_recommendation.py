import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / '..' / 'data' / 'processed' / 'crop_recommendation_processed.csv'
RAW_DATA_PATH = BASE_DIR / '..' / 'data' / 'raw' / 'crop_recommendation.csv'
MODEL_PATH = BASE_DIR / '..' / 'models' / 'crop_recommendation_model.pkl'
PREDICTIONS_PATH = BASE_DIR / '..' / 'outputs' / 'predictions' / 'crop_recommendations.csv'

# Resolve absolute paths
DATA_PATH = DATA_PATH.resolve()
RAW_DATA_PATH = RAW_DATA_PATH.resolve()
MODEL_PATH = MODEL_PATH.resolve()
PREDICTIONS_PATH = PREDICTIONS_PATH.resolve()

# Load raw data to extract crop names
try:
    raw_df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='skip')
except FileNotFoundError:
    raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}. Ensure the raw dataset is available.")

# Create crop ID-to-name mapping (for numeric labels)
unique_crops = sorted(raw_df['label'].unique())
crop_id_to_name = {i: name for i, name in enumerate(unique_crops)}
crop_name_to_id = {name: i for i, name in enumerate(unique_crops)}

# Load processed data
try:
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
except FileNotFoundError:
    raise FileNotFoundError(f"Processed data not found at {DATA_PATH}. Run preprocessing first.")

print("DataFrame Info (Processed):")
print(df.info())
print("\nColumns:", df.columns.tolist())
print("\nNaN Count:\n", df.isna().sum())
print("\nUnique Crop Labels (from processed):\n", df['label'].unique().tolist())

# Select features and target
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
if not all(col in df.columns for col in feature_cols):
    raise ValueError(f"Missing features: {set(feature_cols) - set(df.columns)}")

X = df[feature_cols].copy()
y = df['label']

# --- FIX: Force all labels to crop names ---
# If labels are numeric (e.g., 0, 1, 2...), map them back to names using the raw crop mapping
if np.issubdtype(y.dtype, np.number):
    print("\nDetected numeric crop labels — mapping back to names using raw data.")
    y = y.map(crop_id_to_name)

# Label encode text crop names
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded Labels:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Handle missing values
X = X.fillna(X.mean())
y = pd.Series(y).fillna(pd.Series(y).mode()[0])

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
preds_decoded = label_encoder.inverse_transform(preds)
y_test_decoded = label_encoder.inverse_transform(y_test)

accuracy = accuracy_score(y_test_decoded, preds_decoded)
print("\n✅ Test Set Accuracy:", f"{accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test_decoded, preds_decoded))

# Save model
os.makedirs(MODEL_PATH.parent, exist_ok=True)
joblib.dump({'model': model, 'label_encoder': label_encoder}, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")

# Save predictions
os.makedirs(PREDICTIONS_PATH.parent, exist_ok=True)
pd.DataFrame({'Actual': y_test_decoded, 'Predicted': preds_decoded}).to_csv(PREDICTIONS_PATH, index=False)
print(f"📄 Predictions saved to: {PREDICTIONS_PATH}")

# Function for user prediction
def predict_user_input(model, feature_cols, label_encoder):
    print("\n🌾 Enter the following details to get a crop recommendation:")
    user_input = {}
    feature_ranges = {
        'N': (0, 140, "Nitrogen (kg/ha)"),
        'P': (5, 145, "Phosphorus (kg/ha)"),
        'K': (5, 205, "Potassium (kg/ha)"),
        'temperature': (10, 40, "Temperature (°C)"),
        'humidity': (20, 100, "Humidity (%)"),
        'ph': (4, 9, "pH (4-9)"),
        'rainfall': (20, 300, "Rainfall (mm)")
    }

    for feature in feature_cols:
        min_val, max_val, desc = feature_ranges[feature]
        while True:
            try:
                value = float(input(f"Enter {desc}: "))
                if min_val <= value <= max_val:
                    user_input[feature] = value
                    break
                else:
                    print(f"⚠️ Please enter a value between {min_val} and {max_val}.")
            except ValueError:
                print("⚠️ Please enter a valid number.")

    input_df = pd.DataFrame([user_input], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    print(f"\n✅ Recommended Crop: {prediction_label}")

# Run prediction from user input
predict_user_input(model, feature_cols, label_encoder)
