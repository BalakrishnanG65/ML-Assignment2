from sklearn.tree import DecisionTreeClassifier
import pickle, os

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/decision_tree.pkl", "wb"))