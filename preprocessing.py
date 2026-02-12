import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    df = df.copy()

    # Remove rows with NaN values
    df = df.dropna()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop("y", axis=1)
    y = df["y"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y