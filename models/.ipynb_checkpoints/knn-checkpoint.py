from sklearn.neighbors import KNeighborsClassifier
import pickle, os

def train_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/knn.pkl", "wb"))