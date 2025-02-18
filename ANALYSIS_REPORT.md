# Analysis Report for RGrid Machine Learning Challenge

## Introduction

This report presents an analysis of the machine learning challenge. It covers:

- Data exploration
- Model selection
- Evaluation metrics and validation approaches
- Design considerations
- Experimental results

## Data Exploration

- **Variable Input Lengths:**  
  The `description` field contains text entries of varying lengths. Feature extraction (e.g., TF-IDF) must therefore account for differences in document length and word frequency.

- **Label Distribution:**  
  The dataset is roughly balanced across five classes:
  - **Dementia:** 368 samples
  - **ALS:** 368 samples
  - **Obsessive Compulsive Disorder:** 358 samples
  - **Scoliosis:** 335 samples
  - **Parkinson’s Disease:** 330 samples

## Model Selection

Two models have been evaluated:
- **Logistic Regression:**  
  Serves as a strong baseline for high-dimensional sparse data.
- **Random Forest:**  
  Captures non-linear relationships and can model complex interactions.

## Evaluation Metrics & Validation

The project employs:
- **Weighted F1 Score:**  
  To measure class-specific performance while accounting for class imbalance.
- **Precision-Recall AUC (PR-AUC):**  
  To assess performance on imbalanced data.

Two strategies are used:
- **Standard Train/Test Split:**  
  Provides a quick performance snapshot.
- **K-Fold Cross Validation:**  
  Yields robust metrics across different data splits.

## Design Considerations

- **Preprocessing:**  
  TF-IDF is chosen for its simplicity and effectiveness with variable-length text.
- **Modular Design:**  
  Functions for argument parsing, training, evaluation, and validation are separated following SOLID principles.
- **Metric Diversity:**  
  Using multiple evaluation metrics ensures a comprehensive assessment of model performance.

## Experimental Results

**K-Fold Validation Results:**

- Fold 1: F1 = 0.9317, PR-AUC = 0.9790  
- Fold 2: F1 = 0.9345, PR-AUC = 0.9837  
- Fold 3: F1 = 0.9261, PR-AUC = 0.9748  
- Fold 4: F1 = 0.9084, PR-AUC = 0.9773  
- Fold 5: F1 = 0.9377, PR-AUC = 0.9823  

Mean F1 = 0.9277  
Mean PR-AUC = 0.9794  

**Standard Train/Test Split Results:**  


```
                               precision    recall  f1-score   support
                          ALS       0.88      0.90      0.89        73
                     Dementia       0.97      0.99      0.98        67
Obsessive Compulsive Disorder       0.96      0.96      0.96        79
          Parkinson’s Disease       0.92      0.92      0.92        66
                    Scoliosis       0.92      0.88      0.90        67
                     accuracy                           0.93       352
                    macro avg       0.93      0.93      0.93       352
                 weighted avg       0.93      0.93      0.93       352
```

Training complete. Model and vectorizer have been saved.

## Conclusion

- **Preprocessing needs:**  
  Differences in input lengths require robust text feature extraction.
  
- **Model insights:**  
  Logistic Regression provides a reliable baseline, while Random Forest explores non-linear patterns.
  
- **Evaluation strategy:**  
  K-fold validation coupled with multiple metrics (F1 and PR-AUC) confirms strong generalization.
  
This structured analysis supports the chosen design and reinforces the effectiveness of our approach in modeling the data.


