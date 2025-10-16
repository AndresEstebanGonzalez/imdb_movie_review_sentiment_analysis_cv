# 🎬 IMDB Sentiment Analysis with Cross-Validation

Builds a sentiment analysis pipeline using **Logistic Regression** and **TF-IDF vectorization**, evaluated with **5-fold Stratified Cross-Validation**.

---

## 📚 Overview
This project measures model performance stability through cross-validation rather than a single train/test split.  
It provides insight into model generalization across different subsets of the IMDB dataset.

---

## ⚙️ Requirements
pip install pandas scikit-learn

---

## 🧠 Features
- Cleans text (HTML removal, punctuation stripping)
- TF-IDF vectorization (20k features, unigrams + bigrams)
- Logistic Regression model
- 5-fold **StratifiedKFold** cross-validation
- Reports mean ± std for accuracy, F1, precision, and recall

---

## 🚀 Usage
1. Download `data/IMDB Dataset.csv`
2. Run:
   python imdb_sentiment_cv.py

---

## 📊 Example Output
Accuracy   mean±std: 0.8836 ± 0.0041  -> [0.8812 0.8883 0.8845 0.8798 0.8842]
F1         mean±std: 0.8830 ± 0.0038  -> [0.8801 0.8872 0.8839 0.8790 0.8850]
Precision  mean±std: 0.8855 ± 0.0043  -> [0.8821 0.8899 0.8866 0.8804 0.8885]
Recall     mean±std: 0.8832 ± 0.0040  -> [0.8802 0.8868 0.8841 0.8795 0.8856]

---

## 🧩 Next Steps
- Add multiple models to compare via cross-validation  
- Include ROC-AUC and confusion matrix visualizations  
- Save results to a CSV for experiment tracking  
