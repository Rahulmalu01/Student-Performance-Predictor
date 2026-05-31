"""
Visualizer Utilities
Generates all Plotly charts for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ── Color Palette ─────────────────────────────────────────────
GRADE_COLORS = {
    'A+': '#00C853', 'A': '#69F0AE', 'B': '#40C4FF',
    'C': '#FFD740', 'D': '#FF6D00', 'F': '#FF1744'
}

PALETTE = ['#6C63FF', '#00D4AA', '#FF6B6B', '#FFD93D', '#4ECDC4', '#45B7D1']

CHART_TEMPLATE = 'plotly_dark'
BG_COLOR = 'rgba(0,0,0,0)'
PAPER_BG = 'rgba(0,0,0,0)'
FONT_COLOR = '#E0E0E0'
GRID_COLOR = 'rgba(255,255,255,0.08)'


def _base_layout(**kwargs):
    return dict(
        template=CHART_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG_COLOR,
        font=dict(family='Inter, sans-serif', color=FONT_COLOR),
        margin=dict(l=20, r=20, t=40, b=20),
        **kwargs
    )


# ── Grade Distribution ─────────────────────────────────────────

def grade_distribution_pie(df):
    grade_counts = df['final_grade'].value_counts().reset_index()
    grade_counts.columns = ['Grade', 'Count']
    colors = [GRADE_COLORS.get(g, '#888') for g in grade_counts['Grade']]

    fig = go.Figure(go.Pie(
        labels=grade_counts['Grade'],
        values=grade_counts['Count'],
        marker=dict(colors=colors, line=dict(color='#1a1a2e', width=2)),
        hole=0.45,
        textinfo='label+percent',
        textfont=dict(size=13),
        hovertemplate='<b>Grade %{label}</b><br>Students: %{value}<br>Share: %{percent}<extra></extra>',
    ))
    fig.update_layout(**_base_layout(title=dict(text='Grade Distribution', font=dict(size=16))))
    return fig


def grade_distribution_bar(df):
    grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
    grade_counts = df['final_grade'].value_counts().reindex(grade_order, fill_value=0).reset_index()
    grade_counts.columns = ['Grade', 'Count']
    colors = [GRADE_COLORS.get(g, '#888') for g in grade_counts['Grade']]

    fig = go.Figure(go.Bar(
        x=grade_counts['Grade'],
        y=grade_counts['Count'],
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=grade_counts['Count'],
        textposition='outside',
        hovertemplate='<b>Grade %{x}</b><br>Count: %{y}<extra></extra>',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Students per Grade', font=dict(size=16)),
        xaxis=dict(title='Grade', gridcolor=GRID_COLOR),
        yaxis=dict(title='Number of Students', gridcolor=GRID_COLOR),
        bargap=0.3,
    ))
    return fig


# ── CGPA Distribution ─────────────────────────────────────────

def cgpa_histogram(df):
    fig = go.Figure(go.Histogram(
        x=df['final_cgpa'],
        nbinsx=30,
        marker=dict(
            color=df['final_cgpa'],
            colorscale='Viridis',
            line=dict(color='rgba(255,255,255,0.1)', width=0.5)
        ),
        hovertemplate='CGPA: %{x:.1f}<br>Count: %{y}<extra></extra>',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='CGPA Distribution', font=dict(size=16)),
        xaxis=dict(title='CGPA (0–10)', gridcolor=GRID_COLOR),
        yaxis=dict(title='Number of Students', gridcolor=GRID_COLOR),
    ))
    return fig


# ── Attendance vs CGPA Scatter ────────────────────────────────

def attendance_vs_cgpa_scatter(df, sample=400):
    sample_df = df.sample(min(sample, len(df)), random_state=42)
    colors = [GRADE_COLORS.get(g, '#888') for g in sample_df['final_grade']]

    fig = go.Figure(go.Scatter(
        x=sample_df['attendance_pct'],
        y=sample_df['final_cgpa'],
        mode='markers',
        marker=dict(
            color=colors, size=6, opacity=0.7,
            line=dict(color='rgba(255,255,255,0.1)', width=0.5)
        ),
        text=sample_df['final_grade'],
        hovertemplate='Attendance: %{x:.1f}%<br>CGPA: %{y:.2f}<br>Grade: %{text}<extra></extra>',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Attendance vs CGPA', font=dict(size=16)),
        xaxis=dict(title='Attendance (%)', gridcolor=GRID_COLOR),
        yaxis=dict(title='Final CGPA', gridcolor=GRID_COLOR),
    ))
    return fig


# ── Feature Importance ────────────────────────────────────────

def feature_importance_chart(importance_dict, top_n=15):
    from utils.preprocessor import FEATURE_DISPLAY_NAMES
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [FEATURE_DISPLAY_NAMES.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    colors = [f'rgba(108,99,255,{0.4 + 0.6*(v/max(values))})' for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=features[::-1],
        orientation='h',
        marker=dict(color=colors[::-1], line=dict(color='rgba(255,255,255,0.05)', width=1)),
        text=[f'{v*100:.1f}%' for v in values[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f'Top {top_n} Feature Importances (Random Forest)', font=dict(size=16)),
        xaxis=dict(title='Importance Score', gridcolor=GRID_COLOR),
        yaxis=dict(automargin=True, gridcolor=GRID_COLOR),
        height=500,
    ))
    return fig


# ── Model Comparison Bar ──────────────────────────────────────

def model_comparison_chart(results_dict):
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] * 100 for m in models]
    f1_scores = [results_dict[m]['f1_score'] * 100 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Accuracy (%)', x=models, y=accuracies,
        marker_color='#6C63FF', text=[f'{a:.1f}%' for a in accuracies], textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='F1 Score (%)', x=models, y=f1_scores,
        marker_color='#00D4AA', text=[f'{f:.1f}%' for f in f1_scores], textposition='outside'
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Model Performance Comparison', font=dict(size=16)),
        xaxis=dict(title='Model', gridcolor=GRID_COLOR),
        yaxis=dict(title='Score (%)', gridcolor=GRID_COLOR, range=[0, 110]),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    ))
    return fig


# ── Radar Chart (Student vs Class Average) ───────────────────

def student_vs_average_radar(student_data, df_mean):
    categories = [
        'Attendance %', 'Study Hours', 'Assignment %',
        'Mid-Term Score', 'Participation', 'Sleep Hours'
    ]
    feature_keys = [
        'attendance_pct', 'study_hours_per_week', 'assignment_completion_pct',
        'midterm_score', 'participation_score', 'sleep_hours_per_night'
    ]
    scales = [100, 30, 100, 100, 10, 10]

    student_vals = [min(student_data.get(k, 0) / s * 100, 100) for k, s in zip(feature_keys, scales)]
    avg_vals = [min(df_mean.get(k, 0) / s * 100, 100) for k, s in zip(feature_keys, scales)]

    categories_closed = categories + [categories[0]]
    student_closed = student_vals + [student_vals[0]]
    avg_closed = avg_vals + [avg_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_closed, theta=categories_closed, fill='toself',
        name='Class Average',
        line=dict(color='#00D4AA', width=2),
        fillcolor='rgba(0,212,170,0.15)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=student_closed, theta=categories_closed, fill='toself',
        name='This Student',
        line=dict(color='#6C63FF', width=2),
        fillcolor='rgba(108,99,255,0.25)',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Student vs Class Average', font=dict(size=16)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR),
            bgcolor='rgba(0,0,0,0)',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
    ))
    return fig


# ── Correlation Heatmap ───────────────────────────────────────

def correlation_heatmap(df):
    numeric_cols = [
        'prev_cgpa', 'attendance_pct', 'study_hours_per_week',
        'assignment_completion_pct', 'midterm_score', 'num_backlogs',
        'participation_score', 'library_hours_per_week', 'sleep_hours_per_night',
        'stress_level', 'final_cgpa'
    ]
    labels = [
        'Prev CGPA', 'Attendance', 'Study Hrs', 'Assignment %',
        'Midterm', 'Backlogs', 'Participation', 'Library Hrs',
        'Sleep Hrs', 'Stress', 'Final CGPA'
    ]
    corr = df[numeric_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{y} × %{x}: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Correlation'),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Feature Correlation Matrix', font=dict(size=16)),
        height=520,
        xaxis=dict(side='bottom'),
    ))
    return fig


# ── CGPA by Department ────────────────────────────────────────

def cgpa_by_department(df):
    dept_stats = df.groupby('department')['final_cgpa'].agg(['mean', 'std']).reset_index()
    dept_stats.columns = ['Department', 'Mean CGPA', 'Std']

    fig = go.Figure(go.Bar(
        x=dept_stats['Department'],
        y=dept_stats['Mean CGPA'],
        error_y=dict(type='data', array=dept_stats['Std'], visible=True, color='rgba(255,255,255,0.4)'),
        marker=dict(color=PALETTE[:len(dept_stats)]),
        text=dept_stats['Mean CGPA'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg CGPA: %{y:.2f}<extra></extra>',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Average CGPA by Department', font=dict(size=16)),
        xaxis=dict(title='Department', gridcolor=GRID_COLOR),
        yaxis=dict(title='Average CGPA', range=[0, 11], gridcolor=GRID_COLOR),
        bargap=0.35,
    ))
    return fig


# ── Prediction Gauge ──────────────────────────────────────────

def cgpa_gauge(cgpa_value, max_cgpa=10):
    color = '#00C853' if cgpa_value >= 8 else '#1E88E5' if cgpa_value >= 7 else '#FB8C00' if cgpa_value >= 6 else '#E53935'

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=cgpa_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Predicted CGPA', 'font': {'size': 18, 'color': FONT_COLOR}},
        number={'font': {'size': 40, 'color': color}, 'suffix': ' / 10'},
        gauge={
            'axis': {'range': [0, max_cgpa], 'tickwidth': 1, 'tickcolor': FONT_COLOR},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(255,255,255,0.05)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 5],  'color': 'rgba(229,57,53,0.15)'},
                {'range': [5, 6],  'color': 'rgba(251,140,0,0.15)'},
                {'range': [6, 7],  'color': 'rgba(255,215,64,0.15)'},
                {'range': [7, 8],  'color': 'rgba(30,136,229,0.15)'},
                {'range': [8, 10], 'color': 'rgba(0,200,83,0.15)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': cgpa_value
            },
        }
    ))
    fig.update_layout(**_base_layout(height=280))
    return fig


# ── Grade Probability Bar ─────────────────────────────────────

def grade_probability_bar(prob_dict):
    grade_order = ['F', 'D', 'C', 'B', 'A', 'A+']
    grades = [g for g in grade_order if g in prob_dict]
    probs = [prob_dict[g] * 100 for g in grades]
    colors = [GRADE_COLORS.get(g, '#888') for g in grades]

    fig = go.Figure(go.Bar(
        x=grades, y=probs,
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.05)', width=1)),
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside',
        hovertemplate='Grade %{x}: %{y:.1f}%<extra></extra>',
    ))
    fig.update_layout(**_base_layout(
        title=dict(text='Grade Probability Distribution', font=dict(size=15)),
        xaxis=dict(title='Grade', gridcolor=GRID_COLOR),
        yaxis=dict(title='Probability (%)', range=[0, 110], gridcolor=GRID_COLOR),
        height=280,
    ))
    return fig


# ── Department Distribution ───────────────────────────────────

def performance_by_gender(df):
    gender_grade = df.groupby(['gender', 'final_grade']).size().reset_index(name='Count')
    grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']

    fig = px.bar(
        gender_grade, x='final_grade', y='Count', color='gender',
        barmode='group',
        category_orders={'final_grade': grade_order},
        color_discrete_map={'Male': '#6C63FF', 'Female': '#FF6B9D'},
        template=CHART_TEMPLATE,
    )
    fig.update_layout(**_base_layout(
        title=dict(text='Grade Distribution by Gender', font=dict(size=16)),
        xaxis=dict(title='Grade', gridcolor=GRID_COLOR),
        yaxis=dict(title='Count', gridcolor=GRID_COLOR),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    ))
    return fig


# ── Study Hours Box Plot ──────────────────────────────────────

def study_hours_boxplot(df):
    grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
    colors_list = [GRADE_COLORS.get(g, '#888') for g in grade_order]

    fig = go.Figure()
    for g, c in zip(grade_order, colors_list):
        subset = df[df['final_grade'] == g]['study_hours_per_week']
        if len(subset) > 0:
            fig.add_trace(go.Box(
                y=subset, name=g,
                marker_color=c,
                line_color=c,
                boxmean=True,
                hovertemplate=f'Grade {g}<br>Study Hrs: %{{y:.1f}}<extra></extra>',
            ))
    fig.update_layout(**_base_layout(
        title=dict(text='Study Hours Distribution by Grade', font=dict(size=16)),
        xaxis=dict(title='Grade', gridcolor=GRID_COLOR),
        yaxis=dict(title='Study Hours / Week', gridcolor=GRID_COLOR),
        showlegend=False,
    ))
    return fig
