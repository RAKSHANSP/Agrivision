import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Loading dataset directly from file path
data = pd.read_csv("C:/Users/sprak/OneDrive/Desktop/Agrivision/data/raw/Custom_Crops_yield_Historical_Dataset.csv")

# Dropping missing values
data.dropna(inplace=True)

# Converting crop, state, and district names to lowercase for consistency
data['Crop'] = data['Crop'].str.lower()
data['State Name'] = data['State Name'].str.lower()
data['Dist Name'] = data['Dist Name'].str.lower()

# Selecting features and target
feature_columns = ['N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha',
                   'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm',
                   'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day']
X = data[feature_columns]
y = data['Yield_kg_per_ha']

# Training the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# ✅ Evaluate model performance
y_pred = model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
tolerance = 0.10  # 10% range considered "accurate"
accuracy = np.mean(np.abs((y - y_pred) / y) <= tolerance) * 100

print("✅ Model trained successfully!\n")
print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R² Score: {r2:.3f}")
print(f"🎯 Approximate Accuracy (within ±10%): {accuracy:.2f}%\n")

# Saving the model and scaler
joblib.dump(model, "yield_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Defining feature ranges based on dataset for validation
feature_ranges = {
    'N_req_kg_per_ha': (data['N_req_kg_per_ha'].min(), data['N_req_kg_per_ha'].max()),
    'P_req_kg_per_ha': (data['P_req_kg_per_ha'].min(), data['P_req_kg_per_ha'].max()),
    'K_req_kg_per_ha': (data['K_req_kg_per_ha'].min(), data['K_req_kg_per_ha'].max()),
    'Temperature_C': (data['Temperature_C'].min(), data['Temperature_C'].max()),
    'Humidity_%': (data['Humidity_%'].min(), data['Humidity_%'].max()),
    'pH': (data['pH'].min(), data['pH'].max()),
    'Rainfall_mm': (data['Rainfall_mm'].min(), data['Rainfall_mm'].max()),
    'Wind_Speed_m_s': (data['Wind_Speed_m_s'].min(), data['Wind_Speed_m_s'].max()),
    'Solar_Radiation_MJ_m2_day': (data['Solar_Radiation_MJ_m2_day'].min(), data['Solar_Radiation_MJ_m2_day'].max())
}

# Function to predict yield using historical averages
def predict_yield_by_crop(state_name, dist_name, crop_name):
    state_name = state_name.lower()
    dist_name = dist_name.lower()
    crop_name = crop_name.lower()
    
    if state_name not in data['State Name'].unique():
        print(f"⚠️ State '{state_name}' not found in dataset.")
        return
    state_data = data[data['State Name'] == state_name]
    if dist_name not in state_data['Dist Name'].unique():
        print(f"⚠️ District '{dist_name}' not found in state '{state_name}'.")
        return
    crop_data = state_data[state_data['Dist Name'] == dist_name]
    if crop_name not in crop_data['Crop'].unique():
        print(f"⚠️ Crop '{crop_name}' not found in dataset for {dist_name}, {state_name}.")
        return
    
    filtered_data = crop_data[crop_data['Crop'] == crop_name]
    avg_features = filtered_data[feature_columns].mean()
    avg_features_df = pd.DataFrame([avg_features], columns=feature_columns)
    scaled_input = scaler.transform(avg_features_df)
    predicted_yield = model.predict(scaled_input)[0]
    print(f"🌾 Crop: {crop_name.capitalize()}")
    print(f"📍 Location: {dist_name.capitalize()}, {state_name.capitalize()}")
    print(f"📊 Predicted Average Yield: {predicted_yield:.2f} kg/ha")
    yield_kg_per_acre = predicted_yield / 2.471
    print(f"📊 Predicted Yield: {yield_kg_per_acre:.2f} kg/acre\n")

# Function to predict yield using custom inputs
def predict_yield_custom_inputs(state_name, dist_name, crop_name):
    state_name = state_name.lower()
    dist_name = dist_name.lower()
    crop_name = crop_name.lower()
    
    if state_name not in data['State Name'].unique():
        print(f"⚠️ State '{state_name}' not found in dataset.")
        return
    state_data = data[data['State Name'] == state_name]
    if dist_name not in state_data['Dist Name'].unique():
        print(f"⚠️ District '{dist_name}' not found in state '{state_name}'.")
        return
    crop_data = state_data[state_data['Dist Name'] == dist_name]
    if crop_name not in crop_data['Crop'].unique():
        print(f"⚠️ Crop '{crop_name}' not found in dataset for {dist_name}, {state_name}.")
        return
    
    print(f"\nEnter values for {crop_name.capitalize()} yield prediction in {dist_name.capitalize()}, {state_name.capitalize()} (leave blank for historical averages):")
    custom_inputs = []
    for feature, (min_val, max_val) in feature_ranges.items():
        while True:
            user_input = input(f"{feature} ({min_val:.2f} to {max_val:.2f}, or 'avg' for average): ").strip()
            if user_input.lower() == 'avg' or user_input == '':
                avg_value = crop_data[crop_data['Crop'] == crop_name][feature].mean()
                custom_inputs.append(avg_value)
                print(f"Using average {feature}: {avg_value:.2f}")
                break
            try:
                value = float(user_input)
                if min_val <= value <= max_val:
                    custom_inputs.append(value)
                    break
                else:
                    print(f"⚠️ Value out of range ({min_val:.2f} to {max_val:.2f}). Try again.")
            except ValueError:
                print("⚠️ Invalid input. Enter a number or 'avg'.")
    
    custom_inputs_df = pd.DataFrame([custom_inputs], columns=feature_columns)
    scaled_input = scaler.transform(custom_inputs_df)
    predicted_yield = model.predict(scaled_input)[0]
    print(f"\n🌾 Crop: {crop_name.capitalize()}")
    print(f"📍 Location: {dist_name.capitalize()}, {state_name.capitalize()}")
    print(f"📊 Predicted Yield: {predicted_yield:.2f} kg/ha")
    yield_kg_per_acre = predicted_yield / 2.471
    print(f"📊 Predicted Yield: {yield_kg_per_acre:.2f} kg/acre\n")

# User interaction loop
while True:
    print("\nOptions:")
    print("1. Predict yield using historical averages")
    print("2. Predict yield using custom inputs")
    print("3. Exit")
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '3':
        print("👋 Exiting... Have a great day!")
        break
    elif choice in ['1', '2']:
        state_input = input("Enter State Name: ").strip()
        dist_input = input("Enter District Name: ").strip()
        crop_input = input("Enter Crop Name: ").strip()
        if choice == '1':
            predict_yield_by_crop(state_input, dist_input, crop_input)
        else:
            predict_yield_custom_inputs(state_input, dist_input, crop_input)
    else:
        print("⚠️ Invalid choice. Enter 1, 2, or 3.")
