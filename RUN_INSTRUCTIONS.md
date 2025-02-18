# Run Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   - To perform a standard train/test split run:
     ```
     python train.py --mode train --model logistic
     ```
     or to try with a random forest:
     ```
     python train.py --mode train --model random_forest
     ```

3. Run k-fold validation:
   ```
   python train.py --mode kfold --model logistic
   ```
   (You can switch to random forest by setting `--model random_forest`)

4. Start the API server:
   ```
   python main.py
   ```
   The API will start on http://127.0.0.1:5000/

5. Test the API:
   - Using curl:
     ```
     curl -X POST -H "Content-Type: application/json" \
     -d '{"description": "Sample test description about Dementia"}' \
     http://127.0.0.1:5000/predict
     ```
   - Or run the provided tester script:
     ```
     python test.py
     ```
