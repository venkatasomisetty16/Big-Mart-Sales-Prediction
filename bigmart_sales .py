import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# Load datasets
# =========================
train_data = pd.read_csv("train_v9rqX0R.csv")
test_data = pd.read_csv("test_AbJTz2l.csv")

# Split features and target
features = train_data.drop("Item_Outlet_Sales", axis=1)
target = train_data["Item_Outlet_Sales"]

# Detect column types
categorical_feats = features.select_dtypes(include="object").columns
numerical_feats = features.select_dtypes(exclude="object").columns

# =========================
# Preprocessing
# =========================
cat_processor = Pipeline([
    ("fillna", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

num_processor = Pipeline([
    ("fillna", SimpleImputer(strategy="mean"))
])

data_transformer = ColumnTransformer([
    ("categorical", cat_processor, categorical_feats),
    ("numerical", num_processor, numerical_feats)
])

# =========================
# Model definition
# =========================
reg_pipeline = Pipeline([
    ("transformer", data_transformer),
    ("model", HistGradientBoostingRegressor(random_state=42))
])

# =========================
# Train
# =========================
reg_pipeline.fit(features, target)

# =========================
# Predict
# =========================
predictions = reg_pipeline.predict(test_data)

# =========================
# Create submission
# =========================
output_df = pd.DataFrame({
    "Item_Identifier": test_data["Item_Identifier"],
    "Outlet_Identifier": test_data["Outlet_Identifier"],
    "Item_Outlet_Sales": predictions
})

output_df.to_csv("submission.csv", index=False)
print("✅ submission.csv generated successfully!")
