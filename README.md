# 🎓 Student Performance Predictor & Analytics Dashboard

An interactive, AI-powered web application designed to help educational institutions analyze, visualize, and predict student academic performance. By leveraging machine learning models trained on academic, behavioral, and socioeconomic data, the system identifies students who may require additional academic support and provides personalized recommendation plans for improvement.

---

## 🎯 Project Objective
Traditional academic evaluation methods rely strictly on past exam results. This project goes beyond that by incorporating behavioral patterns (study hours, attendance, assignment completion, library usage) and socioeconomic background factors (internet access, part-time jobs) to:
1. **Predict future student outcomes** (Letter Grades & CGPA estimates) with high-confidence machine learning algorithms.
2. **Visualize key factors influencing academic success** to assist teachers and counselors in early interventions.
3. **Offer personalized, priority-ranked recommendations** for students to optimize their study habits, sleep, and stress levels.

---

## ✨ Features
*   **📊 Overview Dashboard:** Key performance indicators (KPIs) like pass rates, grade distributions, average CGPA, and interactive scatter/histogram plots of the overall class data.
*   **🔮 Interactive Predict Performance:** input fields for a student's profile (academics, habits, lifestyle, demographics) to instantly get:
    *   Predicted Letter Grade (A+ to F) with a color-coded indicator card.
    *   Predicted CGPA Estimate (0–10 scale) on a sleek gauge chart.
    *   Class-wide grade probability breakdown (confidence level).
    *   A radar chart comparing the student directly against the class average.
*   **📈 Comparative Analytics:** In-depth group comparisons (e.g., impact of internet access, part-time jobs, and family income on CGPA), correlation heatmaps, and department-level trends.
*   **🎯 Feature Importance:** Explains the "why" behind the predictions, showing which variables (like midterm scores or attendance) have the highest impact according to the Random Forest model.
*   **💡 Academic Recommendations:** Priority-ranked, actionable study tips, resources, and wellness advice tailored to the student's weaknesses (e.g., low attendance, lack of sleep, or stress).

---

## 🛠️ Technology Stack
*   **Core UI:** [Streamlit](https://streamlit.io/) (v1.35.0) - A premium dark-themed interactive web interface.
*   **Machine Learning:** [scikit-learn](https://scikit-learn.org/) - RandomForest, GradientBoosting, Support Vector Classifier, Logistic Regression, and K-Nearest Neighbors.
*   **Data Processing:** Pandas, NumPy.
*   **Interactive Visualizations:** Plotly (Express & Graph Objects) with custom glassmorphism overlays and responsive styling.
*   **Model Persistence:** Joblib.

---

## 📁 Project Architecture
```text
performance_predictions/
├── app.py                     # Main Streamlit dashboard application entry point
├── requirements.txt           # Python library dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Files excluded from version control
├── data/
│   ├── generate_dataset.py    # Generates synthetic data for 1200 students
│   └── student_data.csv       # The generated CSV dataset
├── models/
│   ├── trainer.py             # Preprocessing & training pipeline for the 5 ML models
│   ├── predictor.py           # Interface to load saved models and run predictions
│   └── saved/                 # Serialized model binary (.pkl) & metadata
├── utils/
│   ├── preprocessor.py        # Data cleaning, scaling, and feature engineering
│   ├── recommender.py         # Knowledge-base rule engine for recommendations
│   └── visualizer.py          # Custom Plotly chart rendering functions
└── pages/
    ├── dashboard.py           # Class-wide stats and overview charts
    ├── prediction.py          # Interactive prediction form & student result dashboard
    ├── analytics.py           # Demographics impact & correlation analytics
    ├── feature_importance.py  # Model metrics & global feature importances
    └── recommendations.py     # Priority-ranked action plans & learning resource cards
```

---

## 🚀 Setup & Running Guide

### 1. Prerequisites
Ensure you have Python 3.9 to 3.12 installed on your system.

### 2. Clone the Repository
```bash
git clone <your-repo-link>
cd performance_predictions
```

### 3. Install Dependencies
Install all required python packages:
```bash
pip install -r requirements.txt
```

### 4. Run First-Time Setup
Generate the synthetic dataset of 1,200 students and train the 5 machine learning models:
```bash
python data/generate_dataset.py
python models/trainer.py
```
*(Alternatively, you can skip this step; launching the Streamlit app will detect missing models/datasets and prompt you to run setup with a single click.)*

### 5. Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```

The terminal will print a local URL (usually `http://localhost:8501`). Open it in your browser to interact with the application.

---

## 🤖 Models & Performance
The application trains five classification models and automatically selects the highest-accuracy model for predictions:
1.  **Random Forest Classifier**
2.  **Gradient Boosting Classifier**
3.  **Support Vector Machine (RBF kernel)**
4.  **Logistic Regression (Multinomial)**
5.  **K-Nearest Neighbors (Weighted)**

All model comparisons, validation metrics (Accuracy, F1-Score, Cross-Validation score), and confusion matrices are visualised under the **Feature Importance** tab of the dashboard.
