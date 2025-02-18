# Analysis Report for RGrid Machine Learning Challenge

## Introduction
This report presents an analysis of the machine learning challenge. It covers data exploration, model selection, evaluation strategies, and design decisions made throughout the project.

## Data Exploration
- **Variable Input Lengths:** The `description` data varies considerably in length. This suggests that feature extraction (e.g., using TF-IDF) must account for word frequency and sentence structure.
- **Label Distribution:** The dataset exhibits a roughly balanced distribution across five classes, as noted in the README:
  - Dementia: 368 samples
  - ALS: 368 samples
  - Obsessive Compulsive Disorder: 358 samples
  - Scoliosis: 335 samples
  - Parkinson’s Disease: 330 samples

## Model Selection
Two models have been implemented and compared:
- **Logistic Regression:** Serves as a baseline model and is well-suited for high-dimensional sparse data produced by text feature extraction.
- **Random Forest:** Explores non-linear relationships within the data and can capture more complex interactions. This model was added as an option to allow comparison against the baseline.

## Evaluation Metrics and Validation
The project uses the following metrics:
- **Weighted F1 Score:** To account for potential class imbalance.
- **Precision-Recall AUC (PR-AUC):** For evaluating how well the models perform on imbalanced data.

Two evaluation approaches are utilized:
- **Standard Train-Test Split:** Provides a quick assessment.
- **K-Fold Cross Validation:** Offers a more robust metric by evaluating performance across multiple folds.

## Design Considerations
- The decision to use TF-IDF for text preprocessing was based on its effectiveness with variable-length texts and simplicity.
- Modular design with separate functions for argument parsing, model training, and validation supports the SOLID principles, ensuring easier maintainability and testing.
- Evaluation using multiple metrics allows for a more comprehensive understanding of model performance, ensuring that both overall accuracy and performance on individual classes are addressed.

## Results

The evaluation results from the training and k-fold validation experiments are as follows:

**K-Fold Validation Results:**

Fold 1: F1: 0.9317, PR-AUC: 0.9790  
Fold 2: F1: 0.9345, PR-AUC: 0.9837  
Fold 3: F1: 0.9261, PR-AUC: 0.9748  
Fold 4: F1: 0.9084, PR-AUC: 0.9773  
Fold 5: F1: 0.9377, PR-AUC: 0.9823  
Mean F1: 0.9277  
Mean PR-AUC: 0.9794  

**Standard Train/Test Split Results:**

(venv) jyothish@badi-dhanno:~/CascadeProjects/ml-recruitment$ python train.py  
                               precision    recall  f1-score   support  
  
                          ALS       0.88      0.90      0.89        73  
                     Dementia       0.97      0.99      0.98        67  
Obsessive Compulsive Disorder       0.96      0.96      0.96        79  
          Parkinson’s Disease       0.92      0.92      0.92        66  
                    Scoliosis       0.92      0.88      0.90        67  
  
                     accuracy                           0.93       352  
                    macro avg       0.93      0.93      0.93       352  
                 weighted avg       0.93      0.93      0.93       352  
  
Training complete. Model and vectorizer saved.

## Conclusion
The analysis confirms that:
- Text data requires careful preprocessing due to input variability.
- Logistic Regression serves as a strong baseline, while Random Forest may provide non-linear insights.
- Robust evaluation via k-fold validation coupled with multiple metrics (F1 and PR-AUC) is vital for assessing model generalization.


