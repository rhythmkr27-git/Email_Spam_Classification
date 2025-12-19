# 📧 Email Spam Classification using Machine Learning

This project implements an **Email/SMS Spam Classification System** using **Machine Learning** and **Natural Language Processing (NLP)** techniques.
The system classifies messages as **Spam** or **Ham (Not Spam)** using a probability-based algorithm.

---

## 🧠 Project Overview

Spam messages cause security risks, data loss, and inconvenience to users.
Manual filtering is inefficient due to the large volume of messages.
This project uses machine learning to automatically detect spam messages based on their textual content.

The project demonstrates the practical application of:
- Machine Learning
- Probability and Statistics
- Text Processing

---

## 🎯 Objectives

- To build an automated system for spam detection
- To apply probabilistic machine learning techniques
- To understand text preprocessing and feature extraction
- To evaluate model performance using standard metrics

---

## 📊 Dataset Used

- **Name:** SMS Spam Collection Dataset  
- **Source:** Kaggle / UCI Machine Learning Repository  
- **Total Records:** ~5,500 messages  
- **Classes:** Spam, Ham  

### Dataset Format
```
label,message
spam,You have won a free prize!
ham,Please send the assignment today
```

---

## 🔄 Data Preprocessing

- Removal of unnecessary columns
- Renaming columns for consistency
- Lowercasing text
- Removing stopwords
- Converting text to numerical format using TF-IDF

---

## 🔍 Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical vectors.
It assigns higher importance to words that are meaningful and less frequent across messages.

---

## 🤖 Machine Learning Algorithm

**Naive Bayes Classifier**
- Based on Bayes’ Theorem
- Probabilistic approach
- Efficient for text classification
- Widely used in spam filtering systems

---

## 🏗 System Workflow

1. Load dataset  
2. Preprocess text data  
3. Convert text into TF-IDF vectors  
4. Train Naive Bayes classifier  
5. Evaluate model performance  
6. Predict spam or ham for new messages  

---

## 📈 Model Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The trained model achieves approximately **97–98% accuracy**.

---

## 🚀 How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python src/train.py
```

### Run Live Demo
```bash
python demo/demo.py
```

---

## 📌 Applications

Email spam filtering

SMS spam detection

Messaging platforms

Cybersecurity systems

---

## ⚠️ Limitations

Cannot detect completely new spam patterns

Depends on dataset quality

Limited contextual understanding

---

## 🔮 Future Enhancements

Use deep learning models

Build a web or mobile interface

Integrate with real email servers

Multilingual spam detection

