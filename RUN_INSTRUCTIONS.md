
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. (Optional) Run k-fold validation:
   a. Start a Python shell in the project directory.
   b. Run:
      >>> from data_preprocessing import load_data, preprocess_data
      >>> from model_training import k_fold_validation
      >>> data = load_data("data/trials.csv")
      >>> X_tfidf, y, _ = preprocess_data(data)
      >>> k_fold_validation(X_tfidf, y)

4. Start the API:
   python main.py
