import pandas as pd
import numpy as np
import pickle
import requests


# Configuration

MODEL_FILE = "Model.pkl"
INPUT_FILE = "amazon_bidding_27_cols.csv"
OUTPUT_FILE = "predicted_bids.csv"
RAINFOREST_API_KEY = "41CCD23CF84449F6B7F0CA75B4D1DC1B"


# Load Trained Model

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Load Input Data

df = pd.read_csv(INPUT_FILE)


#  Manual Encoding

df['Day_Type'] = df['Day_Type'].map({'Weekday':0, 'Weekend':1})


#  Rainforest API - fetch competitor data

# Example function to get competitor price/rating/review
def fetch_competitor_data(asin):
    url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": RAINFOREST_API_KEY,
        "type": "product",
        "amazon_domain": "amazon.in",
        "asin": asin
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        # Extract competitor info (simplified, example)
        comp_price = float(data['product']['buybox_winner']['price']['value'])
        comp_rating = float(data['product']['rating'])
        comp_review_count = int(data['product']['reviews']['total_reviews'])
        return comp_price, comp_rating, comp_review_count
    except Exception as e:
        print(f" Error fetching ASIN {asin}: {e}")
        # fallback: use original columns
        return np.nan, np.nan, np.nan


#  Optional: update competitor columns with live API

# Make sure df has 'ASIN' column for each product
for i, row in df.iterrows():
    if 'ASIN' in df.columns:
        comp_price, comp_rating, comp_review_count = fetch_competitor_data(row['ASIN'])
        if not np.isnan(comp_price):
            df.loc[i, 'Competitor_Price'] = comp_price
            df.loc[i, 'Comp_Rating'] = comp_rating
            df.loc[i, 'Comp_Review_Count'] = comp_review_count

-
#  Select Features for X

X_columns = [
    'Current_Bid', 'Actual_CPC', 'Suggested_Bid_Min', 'Suggested_Bid_Max',
    'Target_ACOS_Goal','Unit_Price','Competitor_Price','Inventory_Level',
    'Day_Type','My_Rating','Comp_Rating','Comp_Review_Count',
    'Year','Month','Day',
    # Encoded keywords
    'Keyword_aloe drink for health','Keyword_anti-inflammatory supplement',
    'Keyword_ayurvedic insulin support','Keyword_curcumin with piperine',
    'Keyword_eco insulin liquid','Keyword_energy booster drink',
    'Keyword_graviola capsules','Keyword_graviola fruit supplement',
    'Keyword_joint pain relief','Keyword_mixed fruit juice',
    'Keyword_natural detox drink','Keyword_natural diabetic care',
    'Keyword_natural immunity booster','Keyword_organic aloe vera',
    'Keyword_pure aloe juice','Keyword_refreshing fruit punch',
    'Keyword_soursop extract','Keyword_sugar control drink',
    'Keyword_turmeric extract','Keyword_vitamin c drink',
    # Encoded product names
    'Product_Name_Aloe vera juice','Product_Name_Curcumin C3',
    'Product_Name_Eco ensulin','Product_Name_Fruit Drink','Product_Name_Graviola',
    # Encoded Match_Type
    'Match_Type_Broad','Match_Type_Exact','Match_Type_Phrase'
]

X = df[X_columns].values


#  Predict Optimal Bid

df['Optimal_Bid_Predicted'] = model.predict(X)


#  Apply custom bid adjustment logics

for i in range(len(df)):
    bid = df.loc[i, 'Optimal_Bid_Predicted']
    # 1. Confidence boost
    if df.loc[i, 'My_Rating'] > df.loc[i, 'Comp_Rating']:
        bid *= 1.1
    # 2. Price conservative
    if df.loc[i, 'Unit_Price'] > df.loc[i, 'Competitor_Price']:
        bid *= 0.8
    # 3. Stock < 10 → cool down
    if df.loc[i, 'Inventory_Level'] < 10:
        bid *= 0.3
    # 4. Stock high → no change (already handled)
    df.loc[i, 'Optimal_Bid_Predicted'] = round(bid, 2)


#  Save Output

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Predictions saved to {OUTPUT_FILE}")