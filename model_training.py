from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, average_precision_score
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import label_binarize
import numpy as np

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

def save_model(model, vectorizer, model_filepath: str, vectorizer_filepath: str):
    joblib.dump(model, model_filepath)
    joblib.dump(vectorizer, vectorizer_filepath)

def k_fold_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []
    pr_auc_scores = []
    # Ensure y is a numpy array if it's not already
    y_array = np.array(y)
    classes = np.unique(y_array)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        # PR-AUC calculation: binarize y_test and get probability estimates
        y_test_binarized = label_binarize(y_test, classes=classes)
        y_prob = model.predict_proba(X_test)
        pr_auc = average_precision_score(y_test_binarized, y_prob, average='macro')
        pr_auc_scores.append(pr_auc)
        print(f"Fold: F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Mean PR-AUC: {np.mean(pr_auc_scores):.4f}")
