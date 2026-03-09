# Bank Fraud Detection using Machine Learning

This project analyzes a bank transaction dataset to explore machine learning techniques for detecting potentially fraudulent transactions.

## Models Implemented
- Support Vector Machine (SVM)
- Neural Network
- Random Forest
- Hierarchical Clustering

## Workflow
1. Load and preprocess transaction dataset
2. Feature processing and normalization
3. Train multiple ML models
4. Evaluate models using:
   - Confusion matrix
   - ROC curves
   - Classification reports
5. Compare model performance
6. Visualize decision boundaries and clustering patterns

## Dataset Features
The dataset includes transaction-related attributes such as:

- TransactionAmount
- TransactionDate
- TransactionType
- Location
- DeviceID
- MerchantID
- IP Address
- CustomerAge
- TransactionDuration
- LoginAttempts
- AccountBalance

Total dataset size: **2512 transactions with 16 features**

## Results

| Model | Accuracy |
|------|------|
| SVM | ~0.79 |
| Neural Network | ~0.79 |
| Random Forest | ~0.77 |

### Key Observations

- Random Forest provided useful **feature importance analysis**
- Neural networks required more computation but captured complex patterns
- SVM produced comparable classification accuracy

## Visualizations

The project includes:

- Confusion matrices
- ROC curves
- Decision boundary plots
- Neural network training curves
- Random forest feature importance
- Hierarchical clustering dendrogram

## Tools Used

- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
