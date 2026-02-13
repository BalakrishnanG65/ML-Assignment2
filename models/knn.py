from sklearn.neighbors import KNeighborsClassifier
import pickle, os
from sklearn.utils.class_weight import compute_sample_weight

def train_model(X_train, y_train):
    # KNN uses weights parameter during prediction
    # Calculate sample weights for class imbalance
    sample_weight = compute_sample_weight('balanced', y_train)
    
    model = KNeighborsClassifier(
        n_neighbors=7,  # Increased from 5 for better stability
        weights='distance',  # Weight by distance
        metric='minkowski',
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/knn.pkl", "wb"))