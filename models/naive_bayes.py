from sklearn.naive_bayes import GaussianNB
import pickle, os
from imblearn.over_sampling import SMOTE

def train_model(X_train, y_train):
    # Use SMOTE to handle class imbalance by oversampling minority class
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    model = GaussianNB()
    model.fit(X_train_balanced, y_train_balanced)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/naive_bayes.pkl", "wb"))