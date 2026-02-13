from sklearn.tree import DecisionTreeClassifier
import pickle, os

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        max_depth=10,  # Prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/decision_tree.pkl", "wb"))