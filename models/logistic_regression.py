from sklearn.linear_model import LogisticRegression
import pickle, os

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=5000,solver="saga",n_jobs=-1)
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/logistic_regression.pkl", "wb"))