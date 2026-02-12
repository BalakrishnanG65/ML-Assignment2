from sklearn.ensemble import RandomForestClassifier
import pickle, os

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/random_forest.pkl", "wb"))