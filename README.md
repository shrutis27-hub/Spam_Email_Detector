# Spam Email Detector using Machine Learning

This project detects whether an email is Spam or Ham using Machine Learning.

## Problem Statement
Email inboxes are often cluttered with spam, promotional and fraudulent emails.
Manual filtering is inefficient and unreliable.
This project implements an automated Spam Email Detection System using Machine Learning techniques.


##  Proposed Approach
1. Load and explore spam datasets  
2. Data cleaning and preprocessing  
3. Feature extraction using TF-IDF Vectorizer 
4. Model training using different classifiers  
5. Testing on unseen data  
6. Real-time spam prediction through UI  


## Datasets Used
1. **Spam-Ham Dataset**  
   - Contains labeled SMS/email messages  
   - Classes: `spam` and `ham`

2. **Enron Spam Subset**  
   - Real-world corporate email dataset  
   - Includes legitimate (ham) and spam emails  
   - Helps the model generalize better to real inbox data

---

## Technologies Used
- Python  
- Jupyter Notebook  
- Scikit-learn  
- Pandas, NumPy  
- Streamlit (for User Interface)


##  Machine Learning Models Implemented
- Multinomial Naive Bayes  
- Logistic Regression  
- Linear Support Vector Classifier (SVM)

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  


##  Results
The models show good performance on both datasets.



- Integrate with live email systems  
- Multi-language spam detection
