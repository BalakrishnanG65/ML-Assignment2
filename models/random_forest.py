from sklearn.ensemble import RandomForestClassifier
import pickle, os

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        max_depth=15,  # Prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/random_forest.pkl", "wb"))