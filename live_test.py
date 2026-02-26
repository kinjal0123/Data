import pandas as pd
import pickle

# Load trained model
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

# 👇 Live input (no CSV needed)
live_input = {
    'Current_Bid': 32.59,
    'Actual_CPC': 23.51,
    'Suggested_Bid_Min': 16.46,
    'Suggested_Bid_Max': 35.27,
    'Target_ACOS_Goal': 20,
    'Unit_Price': 456,
    'Competitor_Price': 470,
    'Inventory_Level': 216,
    'Day_Type': 'Weekday',
    'My_Rating': 3.8,
    'Comp_Rating': 4.4,
    'Comp_Review_Count': 1061,
    'Year': 2026,
    'Month': 1,
    'Day': 2,

    # One-hot encoded values
    'Keyword_natural detox drink': 1,
    'Product_Name_Aloe vera juice': 1,
    'Match_Type_Exact': 1
}

df = pd.DataFrame([live_input])

# Encode Day_Type
df['Day_Type'] = df['Day_Type'].map({'Weekday':0, 'Weekend':1})

# Fill missing training columns automatically
all_columns = model.feature_names_in_

for col in all_columns:
    if col not in df.columns:
        df[col] = 0

# Arrange in correct order
X = df[all_columns]

# Predict
prediction = model.predict(X)

print("🔥 LIVE MODEL OUTPUT:", prediction[0])