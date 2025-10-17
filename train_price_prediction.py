import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / '..' / 'data' / 'processed' / 'price_data_processed.csv'
MODEL_PATH = BASE_DIR / '..' / 'models' / 'price_prediction_model_{commodity}.pkl'
PREDICTIONS_PATH = BASE_DIR / '..' / 'outputs' / 'predictions' / 'price_predictions_{commodity}.csv'
FUTURE_PREDICTIONS_PATH = BASE_DIR / '..' / 'outputs' / 'predictions' / 'future_price_predictions.csv'
SUMMARY_PATH = BASE_DIR / '..' / 'outputs' / 'price_prediction_summary.csv'

DATA_PATH = DATA_PATH.resolve()

# Load data
try:
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
except FileNotFoundError:
    raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run src/data_preprocessing.py first.")

# Preprocess
df['month'] = pd.to_datetime(df['month'], errors='coerce')
df = df.sort_values('month')
if 'avg_modal_price' not in df.columns or 'commodity_name' not in df.columns:
    raise ValueError("Required columns 'avg_modal_price' or 'commodity_name' not found.")

# Function to create lagged features
def create_features(data, lag=3):
    df = data.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['avg_modal_price'].shift(i)
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    return df

# Function to predict prices for a commodity
def predict_crop_price(commodity, forecast_months=12, train_ratio=0.8):
    crop_df = df[df['commodity_name'] == commodity][['month', 'avg_modal_price']].copy()
    if crop_df.empty:
        print(f"No data for {commodity}, skipping.")
        return None, None, None, None
    
    crop_df = create_features(crop_df, lag=3)
    crop_df = crop_df.dropna()
    
    feature_cols = [col for col in crop_df.columns if col not in ['month', 'avg_modal_price']]
    X = crop_df[feature_cols]
    y = crop_df['avg_modal_price']
    
    train_size = int(len(crop_df) * train_ratio)
    if train_size < 10 or len(crop_df) - train_size < 2:
        print(f"Insufficient data for {commodity}, skipping.")
        return None, None, None, None
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = crop_df['month'][train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # ✅ Approximate accuracy for regression (within ±10% tolerance)
    tolerance = 0.10
    accuracy = np.mean(np.abs((y_test - preds) / y_test) <= tolerance) * 100

    print(f"Commodity: {commodity}, MAE: {mae:.2f}, R²: {r2:.2f}, Approx Accuracy (±10%): {accuracy:.2f}%")
    
    # Forecast future months
    last_data = crop_df.tail(3).copy()
    future_preds = []
    future_dates = pd.date_range(start=crop_df['month'].iloc[-1], periods=forecast_months + 1, freq='ME')[1:]
    
    for _ in range(forecast_months):
        last_features = create_features(last_data, lag=3).tail(1)[feature_cols]
        last_features_scaled = scaler.transform(last_features)
        next_pred = model.predict(last_features_scaled)[0]
        future_preds.append(next_pred)
        new_row = pd.DataFrame({
            'month': [last_data['month'].iloc[-1] + pd.offsets.MonthEnd(1)],
            'avg_modal_price': [next_pred]
        })
        last_data = pd.concat([last_data, new_row], ignore_index=True)
    
    # Save model and scaler
    model_path = str(MODEL_PATH).format(commodity=commodity.replace(' ', '_'))
    os.makedirs(Path(model_path).parent, exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, model_path)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test,
        'Predicted': preds
    })
    predictions_path = str(PREDICTIONS_PATH).format(commodity=commodity.replace(' ', '_'))
    os.makedirs(Path(predictions_path).parent, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    
    return mae, r2, future_preds, future_dates

# Process all commodities
results = []
future_forecasts = []
commodities = df['commodity_name'].unique()
for commodity in commodities:
    print(f"\nProcessing {commodity}...")
    mae, r2, forecast_prices, forecast_dates = predict_crop_price(commodity)
    if mae is not None:
        results.append({
            'Commodity': commodity,
            'MAE': mae,
            'R2': r2
        })
        future_forecasts.append(pd.DataFrame({
            'Commodity': [commodity] * len(forecast_dates),
            'Date': forecast_dates,
            'Forecasted_Price': forecast_prices
        }))

# Save summary
summary_df = pd.DataFrame(results)
os.makedirs(SUMMARY_PATH.parent, exist_ok=True)
summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"\nSaved summary to {SUMMARY_PATH}")

# Save future forecasts
if future_forecasts:
    future_forecasts_df = pd.concat(future_forecasts, ignore_index=True)
    os.makedirs(FUTURE_PREDICTIONS_PATH.parent, exist_ok=True)
    future_forecasts_df.to_csv(FUTURE_PREDICTIONS_PATH, index=False)
    print(f"Saved future forecasts to {FUTURE_PREDICTIONS_PATH}")
    
    # Print forecasts for 2026-01-31
    for commodity in commodities:
        forecast_df = future_forecasts_df[future_forecasts_df['Commodity'] == commodity]
        jan_2026 = forecast_df[forecast_df['Date'] == '2026-01-31']
        if not jan_2026.empty:
            price = jan_2026['Forecasted_Price'].iloc[0]
            print(f"{commodity} price on 2026-01-31: {price:.2f} INR/quintal")
