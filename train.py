import argparse
import joblib
import numpy as np
from data_preprocessing import load_data, preprocess_data, split_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "kfold"], default="train",
                        help="Select mode: train a model or run k-fold validation")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic",
                        help="Select model: logistic or random_forest")
    return parser.parse_args()

# Return the desired model instance
def get_model(model_type: str):
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)
    else:
        return RandomForestClassifier(n_estimators=100, random_state=42)

# Run k-fold validation separately
def run_kfold(args, X_tfidf, y, k=5):
    print("Running k-fold validation...")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []
    pr_auc_scores = []
    y_array = np.array(y)
    classes = np.unique(y_array)
    for train_index, test_index in kf.split(X_tfidf):
        X_train_cv, X_test_cv = X_tfidf[train_index], X_tfidf[test_index]
        y_train_cv, y_test_cv = y_array[train_index], y_array[test_index]
        model = get_model(args.model)
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        f1 = f1_score(y_test_cv, y_pred_cv, average='weighted')
        f1_scores.append(f1)
        # PR-AUC calculation
        y_test_cv_bin = label_binarize(y_test_cv, classes=classes)
        y_prob_cv = model.predict_proba(X_test_cv)
        pr_auc = average_precision_score(y_test_cv_bin, y_prob_cv, average='macro')
        pr_auc_scores.append(pr_auc)
        print(f"Fold: F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Mean PR-AUC: {np.mean(pr_auc_scores):.4f}")

# Run standard training and save the model and vectorizer
def run_training(args, X_tfidf, y, vectorizer):
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)
    model = get_model(args.model)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Training complete. Model and vectorizer saved.")

def main():
    args = parse_args()
    data = load_data('data/trials.csv')
    X_tfidf, y, vectorizer = preprocess_data(data)
    
    if args.mode == "kfold":
        run_kfold(args, X_tfidf, y)
    else:
        run_training(args, X_tfidf, y, vectorizer)

if __name__ == "__main__":
    main()
