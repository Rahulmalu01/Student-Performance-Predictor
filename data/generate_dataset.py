"""
Synthetic Student Dataset Generator
Generates a realistic dataset of 1200 students with academic,
behavioral, and socioeconomic features.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_student_dataset(n_students=1200):
    """Generate a comprehensive synthetic student dataset."""

    # --- Demographic Features ---
    genders = np.random.choice(['Male', 'Female'], size=n_students, p=[0.52, 0.48])
    departments = np.random.choice(
        ['Computer Science', 'Electronics', 'Mechanical', 'Civil', 'Business'],
        size=n_students, p=[0.25, 0.20, 0.20, 0.15, 0.20]
    )
    semesters = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_students)
    age = np.random.randint(18, 25, size=n_students)
    income_bracket = np.random.choice(
        ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
        size=n_students, p=[0.15, 0.20, 0.30, 0.20, 0.15]
    )
    internet_access = np.random.choice([0, 1], size=n_students, p=[0.15, 0.85])
    part_time_job = np.random.choice([0, 1], size=n_students, p=[0.70, 0.30])
    first_gen_student = np.random.choice([0, 1], size=n_students, p=[0.60, 0.40])

    # --- Academic Features ---
    # Previous CGPA (8.5–10 for strong students, 5–8.5 for average)
    prev_cgpa_raw = np.random.normal(loc=7.2, scale=1.3, size=n_students)
    prev_cgpa = np.clip(prev_cgpa_raw, 4.0, 10.0)

    # Attendance % (correlated with CGPA)
    attendance_base = 60 + (prev_cgpa - 4.0) * 5 + np.random.normal(0, 8, n_students)
    attendance = np.clip(attendance_base, 20, 100)

    # Study hours per week
    study_hours_base = 5 + (prev_cgpa - 4.0) * 2 + np.random.normal(0, 3, n_students)
    study_hours = np.clip(study_hours_base, 0, 30)

    # Assignment completion rate (%)
    assignment_base = 50 + (attendance - 50) * 0.6 + np.random.normal(0, 10, n_students)
    assignment_completion = np.clip(assignment_base, 0, 100)

    # Mid-term score (out of 100)
    midterm_base = 40 + (prev_cgpa - 4.0) * 8 + study_hours * 1.2 + np.random.normal(0, 10, n_students)
    midterm_score = np.clip(midterm_base, 0, 100)

    # Number of backlogs
    backlog_prob = np.clip(1 - (prev_cgpa - 4.0) / 7, 0.02, 0.6)
    num_backlogs = np.array([np.random.poisson(lam=p * 3) for p in backlog_prob])
    num_backlogs = np.clip(num_backlogs, 0, 8)

    # --- Behavioral Features ---
    # Class participation score (1–10)
    participation_base = 3 + (attendance / 100) * 5 + np.random.normal(0, 1.5, n_students)
    participation_score = np.clip(participation_base, 1, 10)

    # Library usage hours per week
    library_hours_base = 1 + study_hours * 0.3 + np.random.normal(0, 1.5, n_students)
    library_hours = np.clip(library_hours_base, 0, 20)

    # Extracurricular activities (number of activities 0–5)
    extracurricular = np.random.randint(0, 6, size=n_students)

    # Sleep hours per night
    sleep_hours_base = 7 - part_time_job * 0.8 + np.random.normal(0, 1, n_students)
    sleep_hours = np.clip(sleep_hours_base, 4, 10)

    # Stress level (1–10, inverse correlated with sleep & participation)
    stress_base = 10 - sleep_hours * 0.6 - participation_score * 0.3 + np.random.normal(0, 1.5, n_students)
    stress_level = np.clip(stress_base, 1, 10)

    # Teacher interaction frequency (0 = Never, 1 = Sometimes, 2 = Often)
    teacher_interaction = np.random.choice([0, 1, 2], size=n_students, p=[0.20, 0.50, 0.30])

    # --- Compute Final CGPA ---
    final_cgpa = (
        prev_cgpa * 0.30
        + (attendance / 100) * 1.5
        + study_hours * 0.08
        + (assignment_completion / 100) * 1.0
        + (midterm_score / 100) * 2.5
        + participation_score * 0.07
        + (library_hours * 0.05)
        + (sleep_hours - 6) * 0.05
        - num_backlogs * 0.25
        - stress_level * 0.05
        - part_time_job * 0.1
        + internet_access * 0.1
        + np.random.normal(0, 0.3, n_students)
    )
    final_cgpa = np.clip(final_cgpa, 0, 10)

    # --- Derive Letter Grade ---
    def cgpa_to_grade(cgpa):
        if cgpa >= 9.0:
            return 'A+'
        elif cgpa >= 8.0:
            return 'A'
        elif cgpa >= 7.0:
            return 'B'
        elif cgpa >= 6.0:
            return 'C'
        elif cgpa >= 5.0:
            return 'D'
        else:
            return 'F'

    final_grade = [cgpa_to_grade(c) for c in final_cgpa]
    performance_category = ['Pass' if g not in ['F'] else 'Fail' for g in final_grade]

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        'student_id': [f'STU{str(i+1001).zfill(5)}' for i in range(n_students)],
        'age': age,
        'gender': genders,
        'department': departments,
        'semester': semesters,
        'income_bracket': income_bracket,
        'internet_access': internet_access,
        'part_time_job': part_time_job,
        'first_gen_student': first_gen_student,
        'prev_cgpa': np.round(prev_cgpa, 2),
        'attendance_pct': np.round(attendance, 1),
        'study_hours_per_week': np.round(study_hours, 1),
        'assignment_completion_pct': np.round(assignment_completion, 1),
        'midterm_score': np.round(midterm_score, 1),
        'num_backlogs': num_backlogs,
        'participation_score': np.round(participation_score, 1),
        'library_hours_per_week': np.round(library_hours, 1),
        'extracurricular_activities': extracurricular,
        'sleep_hours_per_night': np.round(sleep_hours, 1),
        'stress_level': np.round(stress_level, 1),
        'teacher_interaction': teacher_interaction,
        'final_cgpa': np.round(final_cgpa, 2),
        'final_grade': final_grade,
        'performance': performance_category
    })

    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_student_dataset(1200)

    output_path = os.path.join('data', 'student_data.csv')
    df.to_csv(output_path, index=False)

    print(f"[OK] Dataset generated: {len(df)} students")
    print(f"[OK] Saved to: {output_path}")
    print(f"\nGrade Distribution:")
    print(df['final_grade'].value_counts().sort_index())
    print(f"\nCGPA Stats:")
    print(df['final_cgpa'].describe().round(2))
