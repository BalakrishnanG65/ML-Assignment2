from xgboost import XGBClassifier
import pickle, os

def train_model(X_train, y_train):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/xgboost.pkl", "wb"))