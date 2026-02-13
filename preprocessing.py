import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, scaler=None, encoders=None, fit=False):
    """
    Preprocess data for model evaluation.
    
    Args:
        df: Input dataframe
        scaler: Fitted scaler (for test data). If None, a new one is created.
        encoders: Dict of fitted encoders (for test data). If None, new ones are created.
        fit: If True, fits new scalers and encoders on the data.
    
    Returns:
        X, y: Preprocessed features and target
        scaler, encoders: (if fit=True) Fitted scaler and encoders for future use
    """
    df = df.copy()

    # Remove rows with NaN values
    df = df.dropna()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    if fit:
        encoders = {}
        for col in cat_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
    else:
        if encoders is None:
            encoders = {}
            for col in cat_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                encoders[col] = encoder
        else:
            for col in cat_cols:
                if col in encoders:
                    df[col] = encoders[col].transform(df[col])

    X = df.drop("y", axis=1)
    y = df["y"]

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler, encoders
    else:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        return X, y