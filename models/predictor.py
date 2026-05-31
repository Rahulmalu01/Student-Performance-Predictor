"""
Predictor Interface
Loads trained models and provides prediction API.
"""

import numpy as np
import joblib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import StudentDataPreprocessor


GRADE_COLORS = {
    'A+': '#00C853',
    'A':  '#43A047',
    'B':  '#1E88E5',
    'C':  '#FB8C00',
    'D':  '#E53935',
    'F':  '#B71C1C',
}

GRADE_DESCRIPTIONS = {
    'A+': 'Outstanding performance! Top of the class.',
    'A':  'Excellent performance. Well above average.',
    'B':  'Good performance. Above average.',
    'C':  'Average performance. Room for improvement.',
    'D':  'Below average. Significant improvement needed.',
    'F':  'Failing. Immediate intervention required.',
}

CGPA_TO_PERCENTAGE = {
    'A+': (90, 100),
    'A':  (80, 90),
    'B':  (70, 80),
    'C':  (60, 70),
    'D':  (50, 60),
    'F':  (0, 50),
}


class StudentPredictor:
    def __init__(self, models_dir='models/saved'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.metadata = None
        self._load()

    def _load(self):
        """Load all saved models and preprocessor."""
        # Load metadata
        meta_path = os.path.join(self.models_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Load preprocessor
        self.preprocessor = StudentDataPreprocessor.load(self.models_dir)

        # Load models
        model_files = {
            'Random Forest':       'random_forest.pkl',
            'Gradient Boosting':   'gradient_boosting.pkl',
            'Support Vector Machine': 'support_vector_machine.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
        }

        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

    def predict(self, feature_dict, model_name=None):
        """
        Predict grade and CGPA for a single student.

        Returns:
            dict with grade, cgpa_estimate, confidence, probabilities
        """
        if model_name is None:
            model_name = self.metadata['best_model']

        model = self.models[model_name]
        X = self.preprocessor.transform_single(feature_dict)

        # Grade prediction
        grade_encoded = model.predict(X)[0]
        grade = self.preprocessor.decode_grade([grade_encoded])[0]

        # Probability scores
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            classes = self.preprocessor.get_grade_classes()
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
            confidence = float(max(proba)) * 100
        else:
            prob_dict = {grade: 1.0}
            confidence = 85.0

        # CGPA estimate from grade
        cgpa_range = CGPA_TO_PERCENTAGE.get(grade, (50, 70))
        cgpa_mid = np.mean(cgpa_range)
        cgpa_estimate = round((cgpa_mid / 10), 2)  # scale to 10-pt CGPA

        return {
            'grade': grade,
            'cgpa_estimate': cgpa_estimate,
            'confidence': confidence,
            'probabilities': prob_dict,
            'color': GRADE_COLORS.get(grade, '#888888'),
            'description': GRADE_DESCRIPTIONS.get(grade, ''),
            'model_used': model_name,
        }

    def predict_all_models(self, feature_dict):
        """Run prediction with all models for comparison."""
        results = {}
        for name in self.models:
            results[name] = self.predict(feature_dict, model_name=name)
        return results

    def get_model_results(self):
        """Return stored training results for all models."""
        return self.metadata.get('model_results', {})

    def get_feature_importance(self):
        """Return feature importance from Random Forest."""
        return self.metadata.get('feature_importance', {})

    def get_best_model_name(self):
        return self.metadata.get('best_model', 'Random Forest')

    def get_grade_classes(self):
        return self.metadata.get('grade_classes', [])

    def is_loaded(self):
        return len(self.models) > 0 and self.preprocessor is not None
