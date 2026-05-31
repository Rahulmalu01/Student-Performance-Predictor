"""
Data Preprocessor
Handles feature engineering, encoding, and scaling for ML pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


FEATURE_COLUMNS = [
    'age', 'gender', 'department', 'semester', 'income_bracket',
    'internet_access', 'part_time_job', 'first_gen_student',
    'prev_cgpa', 'attendance_pct', 'study_hours_per_week',
    'assignment_completion_pct', 'midterm_score', 'num_backlogs',
    'participation_score', 'library_hours_per_week',
    'extracurricular_activities', 'sleep_hours_per_night',
    'stress_level', 'teacher_interaction'
]

CATEGORICAL_COLS = ['gender', 'department', 'income_bracket']
BINARY_COLS = ['internet_access', 'part_time_job', 'first_gen_student']
ORDINAL_COLS = ['teacher_interaction']

TARGET_GRADE = 'final_grade'
TARGET_CGPA = 'final_cgpa'

GRADE_ORDER = ['F', 'D', 'C', 'B', 'A', 'A+']

INCOME_ORDER = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']

FEATURE_DISPLAY_NAMES = {
    'prev_cgpa': 'Previous CGPA',
    'midterm_score': 'Mid-Term Score',
    'attendance_pct': 'Attendance %',
    'assignment_completion_pct': 'Assignment Completion %',
    'study_hours_per_week': 'Study Hours/Week',
    'participation_score': 'Class Participation Score',
    'num_backlogs': 'Number of Backlogs',
    'stress_level': 'Stress Level',
    'library_hours_per_week': 'Library Usage (hrs/week)',
    'sleep_hours_per_night': 'Sleep Hours/Night',
    'extracurricular_activities': 'Extracurricular Activities',
    'teacher_interaction': 'Teacher Interaction',
    'internet_access': 'Internet Access',
    'part_time_job': 'Part-Time Job',
    'first_gen_student': 'First Generation Student',
    'income_bracket': 'Family Income Bracket',
    'department': 'Department',
    'gender': 'Gender',
    'semester': 'Semester',
    'age': 'Age',
}


class StudentDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.grade_encoder = LabelEncoder()
        self.fitted = False

    def _encode_categoricals(self, df, fit=True):
        df = df.copy()

        # Income bracket ordinal encoding
        income_map = {v: i for i, v in enumerate(INCOME_ORDER)}
        df['income_bracket'] = df['income_bracket'].map(income_map).fillna(2)

        # Gender binary encoding
        df['gender'] = (df['gender'] == 'Female').astype(int)

        # Department one-hot encoding
        if fit:
            dummies = pd.get_dummies(df['department'], prefix='dept')
            self.dept_columns = dummies.columns.tolist()
        else:
            dummies = pd.get_dummies(df['department'], prefix='dept')
            # Align columns
            for col in self.dept_columns:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[self.dept_columns]

        df = df.drop(columns=['department'])
        df = pd.concat([df, dummies], axis=1)

        return df

    def _engineer_features(self, df):
        df = df.copy()
        # Study efficiency index
        df['study_efficiency'] = df['study_hours_per_week'] * (df['assignment_completion_pct'] / 100)
        # Attendance-grade interaction
        df['attendance_x_prev_cgpa'] = df['attendance_pct'] * df['prev_cgpa'] / 100
        # Academic stress index
        df['academic_pressure'] = df['num_backlogs'] * df['stress_level']
        # Wellness score
        df['wellness_score'] = df['sleep_hours_per_night'] - df['stress_level'] * 0.5
        return df

    def fit_transform(self, df):
        """Fit on training data and transform."""
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=True)

        # Select numeric features only
        non_feature = ['student_id', 'final_grade', 'final_cgpa', 'performance']
        feature_cols = [c for c in df.columns if c not in non_feature]
        self.feature_cols = feature_cols

        X = df[feature_cols].values
        X = self.scaler.fit_transform(X)

        y_grade = self.grade_encoder.fit_transform(df[TARGET_GRADE])
        y_cgpa = df[TARGET_CGPA].values

        self.fitted = True
        return X, y_grade, y_cgpa, feature_cols

    def transform(self, df):
        """Transform new data using fitted scaler."""
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=False)
        X = df[self.feature_cols].values
        X = self.scaler.transform(X)
        return X

    def transform_single(self, feature_dict):
        """Transform a single student's data dict."""
        df = pd.DataFrame([feature_dict])
        return self.transform(df)

    def decode_grade(self, encoded):
        return self.grade_encoder.inverse_transform(encoded)

    def get_grade_classes(self):
        return self.grade_encoder.classes_.tolist()

    def save(self, path='models/saved'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'preprocessor.pkl'))

    @staticmethod
    def load(path='models/saved'):
        return joblib.load(os.path.join(path, 'preprocessor.pkl'))


def load_data(filepath='data/student_data.csv'):
    df = pd.read_csv(filepath)
    return df


def get_train_test_split(X, y_grade, y_cgpa, test_size=0.2):
    X_train, X_test, yg_train, yg_test, yc_train, yc_test = train_test_split(
        X, y_grade, y_cgpa, test_size=test_size, random_state=42, stratify=y_grade
    )
    return X_train, X_test, yg_train, yg_test, yc_train, yc_test
