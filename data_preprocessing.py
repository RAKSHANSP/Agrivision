import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define base directories (relative to src/)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / '..' / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / '..' / 'data' / 'processed'

def preprocess_crop_recommendation(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Preprocess Crop_recommendation.csv for crop recommendation model."""
    try:
        df = pd.read_csv(input_path, on_bad_lines='skip')
        expected_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        if not all(col in df.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in df.columns]
            raise ValueError(f"Missing columns in Crop_recommendation.csv: {missing}")
        
        # Fill NaNs with mean for numerics
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Encode label to categorical codes (for model compatibility)
        df['label'] = df['label'].astype('category').cat.codes
        
        # Save
        os.makedirs(output_path.parent, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed crop recommendation data to {output_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File {input_path} not found. Please check {DATA_DIR}")
        raise
    except Exception as e:
        logger.error(f"Error processing crop recommendation data: {e}")
        raise

def preprocess_yield_data(production_path: Path, yield_path: Path, output_path: Path) -> pd.DataFrame:
    """Preprocess crop_production.csv and Custom_Crops_yield_Historical_Dataset.csv for yield prediction."""
    try:
        prod = pd.read_csv(production_path, on_bad_lines='skip')
        yield_df = pd.read_csv(yield_path, on_bad_lines='skip')
        
        # Verify expected columns (adjusted for actual data)
        prod_cols = ['State_Name', 'Crop', 'Crop_Year', 'Production', 'Area']
        yield_cols = ['State Name', 'Crop', 'Year', 'Yield_kg_per_ha', 'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm']
        for col, df, name in [(prod_cols, prod, 'crop_production.csv'), (yield_cols, yield_df, 'Custom_Crops_yield_Historical_Dataset.csv')]:
            if not all(c in df.columns for c in col):
                missing = [c for c in col if c not in df.columns]
                raise ValueError(f"Missing columns in {name}: {missing}")
        
        # Debug: Print unique values of join keys
        logger.info("prod State_Name unique (sample): %s", prod['State_Name'].unique()[:5])
        logger.info("yield_df State Name unique (sample): %s", yield_df['State Name'].unique()[:5])
        logger.info("prod Crop unique (sample): %s", prod['Crop'].unique()[:5])
        logger.info("yield_df Crop unique (sample): %s", yield_df['Crop'].unique()[:5])
        logger.info("prod Crop_Year unique (sample): %s", prod['Crop_Year'].unique()[:5])
        logger.info("yield_df Year unique (sample): %s", yield_df['Year'].unique()[:5])
        
        # Standardize join keys (lower, strip)
        prod['State_Name'] = prod['State_Name'].str.lower().str.strip()
        prod['Crop'] = prod['Crop'].str.lower().str.strip()
        yield_df['State Name'] = yield_df['State Name'].str.lower().str.strip()
        yield_df['Crop'] = yield_df['Crop'].str.lower().str.strip()
        prod['Crop_Year'] = pd.to_numeric(prod['Crop_Year'], errors='coerce')
        yield_df['Year'] = pd.to_numeric(yield_df['Year'], errors='coerce')
        
        # Compute yield from production if not present
        prod['Yield'] = prod['Production'] / prod['Area']
        
        # Improved merge: First, inner on state/crop/year
        merged = prod.merge(
            yield_df, 
            left_on=['State_Name', 'Crop', 'Crop_Year'],
            right_on=['State Name', 'Crop', 'Year'], 
            how='left'  # Use left to keep all prod data, add env where match
        )
        
        # Debug merge
        logger.info(f"Merged DataFrame Shape: {merged.shape}")
        logger.info(f"Merged Columns: {merged.columns.tolist()}")
        
        match_rate = merged.dropna(subset=['State Name']).shape[0] / prod.shape[0]
        logger.info(f"Match rate with yield data: {match_rate:.2%}")
        
        # If low match, fallback: Enrich with average env per state/crop from yield_df
        if match_rate < 0.1:  # Arbitrary threshold
            logger.warning("Low match rate. Enriching with average env per state/crop.")
            # Compute avg env per state/crop from yield_df
            env_avg = yield_df.groupby(['State Name', 'Crop'])[
                ['Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm', 'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day', 'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha']
            ].mean().reset_index()
            
            # Merge on state/crop only
            merged = prod.merge(
                env_avg, 
                left_on=['State_Name', 'Crop'],
                right_on=['State Name', 'Crop'], 
                how='left'
            )
        
        # Ensure Yield is present (from prod)
        if 'Yield' not in merged.columns:
            merged['Yield'] = merged['Production'] / merged['Area']
        
        # Rename yield from yield_df if present and prefer it, else use computed
        if 'Yield_kg_per_ha' in merged.columns:
            merged['Yield'] = merged['Yield_kg_per_ha'].fillna(merged['Yield'])
        
        # Handle missing values
        numeric_cols = merged.select_dtypes(include=['float64', 'int64']).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].mean())
        
        # Drop redundant columns
        merged = merged.drop(columns=['State Name', 'Year'], errors='ignore')
        
        # Save
        os.makedirs(output_path.parent, exist_ok=True)
        merged.to_csv(output_path, index=False)
        logger.info(f"Saved processed yield data to {output_path}")
        return merged
    except FileNotFoundError:
        logger.error(f"File {production_path} or {yield_path} not found in {DATA_DIR}")
        raise
    except Exception as e:
        logger.error(f"Error processing yield data: {e}")
        raise

def preprocess_price_data(current_path: Path, historical_path: Path, output_path: Path) -> pd.DataFrame:
    """Preprocess Crop_price.csv and crop_price_dataset.csv for price prediction."""
    try:
        current = pd.read_csv(current_path, on_bad_lines='skip')
        historical = pd.read_csv(historical_path, on_bad_lines='skip')
        
        # Clean column names in current
        current.columns = current.columns.str.replace('_x0020_', ' ', regex=False)
        
        current_cols = ['Commodity', 'Arrival_Date', 'Modal Price']
        historical_cols = ['month', 'commodity_name', 'avg_modal_price']
        for col, df, name in [(current_cols, current, 'Crop_price.csv'), (historical_cols, historical, 'crop_price_dataset.csv')]:
            if not all(c in df.columns for c in col):
                missing = [c for c in col if c not in df.columns]
                raise ValueError(f"Missing columns in {name}: {missing}")
        
        # Datetime conversion
        historical['month'] = pd.to_datetime(historical['month'], errors='coerce')
        current['Arrival_Date'] = pd.to_datetime(current['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
        
        # Aggregate current to monthly mean by commodity
        current['month'] = current['Arrival_Date'].dt.to_period('M').dt.to_timestamp()
        current_agg = current.groupby(['Commodity', 'month'])['Modal Price'].mean().reset_index()
        current_agg = current_agg.rename(columns={'Commodity': 'commodity_name', 'Modal Price': 'avg_modal_price'})
        
        # Concat with historical
        merged = pd.concat([
            historical[['month', 'commodity_name', 'avg_modal_price']],
            current_agg[['month', 'commodity_name', 'avg_modal_price']]
        ], ignore_index=True)
        
        # Standardize commodity names (lower, strip)
        merged['commodity_name'] = merged['commodity_name'].str.lower().str.strip()
        
        # Fill NaNs
        numeric_cols = merged.select_dtypes(include=['float64', 'int64']).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].mean())
        
        # Sort by date
        merged = merged.sort_values('month').reset_index(drop=True)
        
        # Save
        os.makedirs(output_path.parent, exist_ok=True)
        merged.to_csv(output_path, index=False)
        logger.info(f"Saved processed price data to {output_path}")
        return merged
    except FileNotFoundError:
        logger.error(f"File {current_path} or {historical_path} not found in {DATA_DIR}")
        raise
    except Exception as e:
        logger.error(f"Error processing price data: {e}")
        raise

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    crop_recommend_path = DATA_DIR / 'Crop_recommendation.csv'
    yield_production_path = DATA_DIR / 'crop_production.csv'
    yield_historical_path = DATA_DIR / 'Custom_Crops_yield_Historical_Dataset.csv'
    price_current_path = DATA_DIR / 'Crop_price.csv'
    price_historical_path = DATA_DIR / 'crop_price_dataset.csv'
    
    preprocess_crop_recommendation(
        crop_recommend_path,
        PROCESSED_DIR / 'crop_recommendation_processed.csv'
    )
    preprocess_yield_data(
        yield_production_path,
        yield_historical_path,
        PROCESSED_DIR / 'yield_data_processed.csv'
    )
    preprocess_price_data(
        price_current_path,
        price_historical_path,
        PROCESSED_DIR / 'price_data_processed.csv'
    )
    logger.info("Data preprocessing complete!")