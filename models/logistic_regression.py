from sklearn.linear_model import LogisticRegression
import pickle, os

def train_model(X_train, y_train):
    # Use class_weight='balanced' to handle class imbalance
    # This automatically adjusts weights inversely proportional to class frequencies
    model = LogisticRegression(
        max_iter=5000,
        solver="saga",
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/logistic_regression.pkl", "wb"))