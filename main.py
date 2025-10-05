import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except:
    has_xgb = False

# ------------------------
# Load Data & Train Model
# ------------------------
@st.cache_data
def train_model():
    data = pd.read_csv("train.csv")   # Kaggle House Prices dataset
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    y = data["SalePrice"]

    # Features
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # Best model → Random Forest (could switch to XGBoost if available)
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return pipeline, list(num_features), list(cat_features), rmse, r2, list(X.columns)

pipeline, num_features, cat_features, rmse, r2, all_features = train_model()

# ------------------------
# Streamlit UI
# ------------------------
st.title("🏠 House Price Prediction App")
st.write("Enter house details below and get an estimated **SalePrice** using a trained model.")

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"RMSE: {rmse:.2f}")
st.sidebar.write(f"R²: {r2:.3f}")

# ------------------------
# User Input Form
# ------------------------
st.subheader("Enter House Features")

user_input = {}

# Example: Only ask for a few important features
important_num = ["GrLivArea", "LotArea", "YearBuilt", "1stFlrSF", "GarageArea"]
important_cat = ["Neighborhood", "HouseStyle", "ExterQual", "KitchenQual", "SaleCondition"]

for col in important_num:
    user_input[col] = st.number_input(f"{col}", value=0.0)

for col in important_cat:
    user_input[col] = st.text_input(f"{col}", "")

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# ------------------------
# Fill Missing Columns
# ------------------------
# Add missing columns as NaN
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = np.nan

# Reorder columns
input_df = input_df[all_features]

# ------------------------
# Prediction
# ------------------------
if st.button("🔮 Predict SalePrice"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"🏡 Estimated House Price: **${prediction:,.2f}**")
