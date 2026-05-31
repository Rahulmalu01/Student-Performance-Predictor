"""
ML Model Trainer
Trains 5 classification models and selects the best performer.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import StudentDataPreprocessor, load_data, get_train_test_split


MODELS = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
    ),
    'Support Vector Machine': SVC(
        kernel='rbf', C=10, gamma='scale', probability=True, random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, multi_class='multinomial'
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='euclidean'
    ),
}


def train_all_models():
    print("=" * 60)
    print("  STUDENT PERFORMANCE PREDICTOR — MODEL TRAINING")
    print("=" * 60)

    # Load data
    df = load_data('data/student_data.csv')
    print(f"\n[OK] Loaded dataset: {len(df)} students, {len(df.columns)} features")

    # Preprocess
    preprocessor = StudentDataPreprocessor()
    X, y_grade, y_cgpa, feature_cols = preprocessor.fit_transform(df)

    X_train, X_test, yg_train, yg_test, yc_train, yc_test = get_train_test_split(
        X, y_grade, y_cgpa
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Grade Classes: {preprocessor.get_grade_classes()}")

    os.makedirs('models/saved', exist_ok=True)
    results = {}
    best_model_name = None
    best_accuracy = 0.0

    print("\n" + "-" * 60)
    print("Training Models...")
    print("-" * 60)

    for name, model in MODELS.items():
        print(f"\nTraining: {name}...")
        model.fit(X_train, yg_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(yg_test, y_pred)
        f1 = f1_score(yg_test, y_pred, average='weighted')
        prec = precision_score(yg_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(yg_test, y_pred, average='weighted', zero_division=0)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, yg_train, cv=5, scoring='accuracy')

        results[name] = {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'confusion_matrix': confusion_matrix(yg_test, y_pred).tolist(),
            'classification_report': classification_report(
                yg_test, y_pred,
                target_names=preprocessor.get_grade_classes(),
                output_dict=True
            )
        }

        print(f"   Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   CV Score:  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Save model
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, f'models/saved/{safe_name}.pkl')

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    # Feature importance from Random Forest
    rf_model = MODELS['Random Forest']
    feature_importance = dict(zip(feature_cols, rf_model.feature_importances_.tolist()))

    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'grade_classes': preprocessor.get_grade_classes(),
        'feature_cols': feature_cols,
        'feature_importance': feature_importance,
        'model_results': results,
        'dataset_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    with open('models/saved/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    preprocessor.save('models/saved')

    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name} ({best_accuracy*100:.1f}% accuracy)")
    print(f"All models saved to models/saved/")
    print("=" * 60)

    return results, best_model_name


if __name__ == '__main__':
    train_all_models()
