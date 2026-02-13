from xgboost import XGBClassifier
import pickle, os
from sklearn.utils.class_weight import compute_sample_weight

def train_model(X_train, y_train):
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5
    )
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(model, open("saved_models/xgboost.pkl", "wb"))