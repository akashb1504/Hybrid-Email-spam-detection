# 📧 Email Spam Detection using Hybrid Classification (Random Forest + LSTM)

## 🧩 Objective

Build a spam detection system that:
- Uses **Random Forest** for fast, shallow classification
- Routes low-confidence predictions to a **Deep LSTM** classifier
- Achieves **high accuracy** without compromising **detection speed**
- Implements a **confidence verifier** to control classification flow

---

## 🧠 Model Architecture

### 🔹 Shallow Classifier – Random Forest (RF)
- Input: **TF-IDF vectors** of top 3,000 frequent words
- Output: Spam/Ham prediction + confidence score
- Confidence Threshold: **0.8**
- Accepted if confidence ≥ threshold; else forwarded to LSTM

### 🔹 Deep Classifier – LSTM
- Input: **Padded sequences** using top 30,000 words
- Layers:
  - Embedding → LSTM(64 units)
  - Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
- Trained only on **rejected samples** from RF (using **random undersampling**)

---

## ⚙️ Tools & Libraries

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- Pandas, NumPy  
- NLTK (for text preprocessing)  
- Matplotlib / Seaborn (for visualization)

---

## 📁 Dataset

- 190K+ Spam | Ham Email Dataset for Classification
- Source: [Kaggle](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification)
---

## 📊 Evaluation Summary

| Model            | Accuracy | TPR    | TNR    |
|------------------|----------|--------|--------|
| Random Forest    | 97.8%    | 97.3%  | 98.2%  |
| LSTM on Rejected | 95.7%    | 93.6%  | 98.6%  |

✅ **Only 11.8%** of validation samples were rejected by RF and routed to LSTM  
✅ Final decision = Accepted RF + Accepted LSTM predictions

---

## 🔁 Workflow

1. **Preprocessing**:
   - Clean email text, remove stopwords
   - TF-IDF vectorization for RF
   - Tokenization + Padding for LSTM

2. **Training**:
   - Split: 40% train, 30% val, 30% test
   - Train Random Forest on full TF-IDF dataset
   - Apply **confidence verifier** on RF predictions
   - Route rejected samples to LSTM
   - Train LSTM on balanced subset of rejected data

3. **Evaluation**:
   - Collect metrics for both RF and LSTM independently
   - Merge final results for complete accuracy

---
