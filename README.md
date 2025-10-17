🎬 IMDB Sentiment Analysis with Cross-Validation

Builds a sentiment analysis pipeline using Logistic Regression and TF-IDF vectorization, evaluated with 5-fold Stratified Cross-Validation.
The pipeline is automatically saved for reuse or deployment.

⸻

📚 Overview

This version focuses on evaluating model consistency and generalization through repeated cross-validation rather than a single train/test split.
It provides a reliable estimate of performance stability across different subsets of the IMDB dataset and saves the trained pipeline for future inference or fine-tuning.

⸻

⚙️ Requirements

pip install pandas scikit-learn joblib


⸻

🧠 Features
	•	✅ Text cleaning (HTML & punctuation removal)
	•	✅ TF-IDF vectorization (20k features, unigrams + bigrams)
	•	✅ Logistic Regression classifier (Liblinear solver)
	•	✅ 5-fold StratifiedKFold cross-validation
	•	✅ Reports mean ± std for accuracy, F1, precision, and recall
	•	✅ Saves the complete pipeline to models/imdb_logreg_cv_pipeline.joblib

⸻

🗂️ Project Structure

.
├── data/
│   └── IMDB Dataset.csv
├── models/
│   └── imdb_logreg_cv_pipeline.joblib
└── imdb_sentiment_cv.py


⸻

🚀 Usage
	1.	Download the IMDB dataset → data/IMDB Dataset.csv
	2.	Run the script:

python imdb_sentiment_cv.py


	3.	After execution:
	•	Cross-validation scores will be printed to the console
	•	The trained pipeline will be saved under models/imdb_logreg_cv_pipeline.joblib

⸻

📊 Example Output

Accuracy   mean±std: 0.8836 ± 0.0041  -> [0.8812 0.8883 0.8845 0.8798 0.8842]
F1         mean±std: 0.8830 ± 0.0038  -> [0.8801 0.8872 0.8839 0.8790 0.8850]
Precision  mean±std: 0.8855 ± 0.0043  -> [0.8821 0.8899 0.8866 0.8804 0.8885]
Recall     mean±std: 0.8832 ± 0.0040  -> [0.8802 0.8868 0.8841 0.8795 0.8856]


⸻

🧩 Next Steps
	•	Compare performance with additional classifiers (e.g., Naive Bayes, SVM)
	•	Log metrics to CSV for experiment tracking
	•	Add visualization of cross-validation score distributions
	•	Integrate model loading and evaluation on unseen test data
