from sklearn.naive_bayes import GaussianNB
import pickle, os

def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/naive_bayes.pkl", "wb"))