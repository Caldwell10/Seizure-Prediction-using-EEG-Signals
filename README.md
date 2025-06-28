#  EEG-Based Seizure Prediction using Machine Learning

## Project Overview
Seizures are unpredictable neurological events that can significantly impact patient health and safety. Accurate and timely seizure prediction using EEG (electroencephalogram) data can enable **early interventions**, improve patient care, and even pave the way for **wearable real-time monitoring systems**.

This project aims to:
- Build and compare machine learning models for **seizure prediction** using EEG signals.
- Support **healthy EEG monitoring** to differentiate between normal and pre-seizure brain activity.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn, Matplotlib
- Google Colab

---

## Dataset
- **Source:** [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
- **Description:** EEG recordings from pediatric patients with intractable seizures.
- **Sampling Strategy:** First 1000 rows sampled from each file, balanced between seizure and non-seizure events.

---

## Project Pipeline
1. **Data Preparation**
   - Combined and balanced seizure and non-seizure EEG samples.
   - Handled missing values using **mean imputation**.
   - Applied **data normalization** using StandardScaler.

2. **Data Visualization**
   - EEG signal plotting for 3-channel comparisons (Seizure vs. No-Seizure).
   - Correlation heatmap to analyze channel relationships.

3. **Model Building**
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - XGBoost
   - Logistic Regression

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - Visualization: Confusion Matrices, ROC Curves, Model Performance Comparison Charts

---

##  Model Performance Summary
| Model                | Accuracy | Precision | Recall | F1-Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Random Forest       | 0.90     | 0.90      | 0.90   | 0.90     | 0.96 |
| K-Nearest Neighbors | 0.91     | 0.91      | 0.91   | 0.91     | 0.97 |
| XGBoost             | 0.74     | 0.75      | 0.74   | 0.74     | 0.83 |
| Logistic Regression | 0.51     | 0.51      | 0.51   | 0.51     | 0.52 |

---

## Key Insights
- **Best Performing Model:** KNN achieved the highest accuracy and AUC score.
- **Random Forest** also performed exceptionally well, closely following KNN.
- **XGBoost** showed moderate performance but could improve with further hyperparameter tuning.
- **Logistic Regression** underperformed, suggesting that simple linear models may not effectively capture the complex, non-linear EEG patterns associated with seizures.

---

## Future Work
- Implement **time-frequency domain feature extraction** (e.g., wavelet transforms) for improved signal representation.
- Explore **deep learning models** such as LSTMs and 1D CNNs for sequential EEG data.
- Develop **wearable real-time seizure detection systems** for proactive patient care.
- Deploy as a cloud-based **EEG monitoring and prediction platform** for clinical use.

---

