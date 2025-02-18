# Test API Instructions

1. Start the API server:
   ```
   python main.py
   ```

2. Test the predict endpoint using curl:
   ```
   curl -X POST -H "Content-Type: application/json" \
   -d '{"description": "Sample test description about Dementia"}' \
   http://127.0.0.1:5000/predict
   ```
   - Expected response (example):
     ```json
     {"prediction": "Dementia", "confidence": 0.98}
     ```
     (If the confidence is below 0.60, the API will return `{"prediction": "unsure", "confidence": <value>}`.)

3. Alternatively, run the provided tester script:
   ```
   python test.py
   ```
