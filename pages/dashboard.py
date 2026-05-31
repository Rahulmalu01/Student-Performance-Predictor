"""
Dashboard Page — Overview with KPI cards and key charts.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizer import (
    grade_distribution_pie, grade_distribution_bar,
    cgpa_histogram, attendance_vs_cgpa_scatter,
    cgpa_by_department, performance_by_gender, study_hours_boxplot
)


def render_kpi(label, value, delta=None, icon='📊', color='#6C63FF'):
    delta_html = f'<p style="font-size:12px;color:#aaa;margin:0">{delta}</p>' if delta else ''
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(108,99,255,0.15),rgba(0,212,170,0.08));
                border:1px solid rgba(108,99,255,0.3);border-radius:16px;
                padding:20px 18px;text-align:center;min-height:110px;">
      <div style="font-size:28px;margin-bottom:4px">{icon}</div>
      <p style="font-size:26px;font-weight:700;color:{color};margin:0;line-height:1.1">{value}</p>
      <p style="font-size:12px;color:#b0b0b0;margin:4px 0 0">{label}</p>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)


def show(df: pd.DataFrame):
    st.markdown("## 📊 Student Performance Dashboard")
    st.markdown("<p style='color:#aaa;margin-top:-10px'>Real-time overview of academic performance across all students.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────
    total = len(df)
    avg_cgpa = df['final_cgpa'].mean()
    pass_rate = (df['performance'] == 'Pass').sum() / total * 100
    top_performers = (df['final_grade'].isin(['A+', 'A'])).sum()
    avg_attendance = df['attendance_pct'].mean()
    avg_study = df['study_hours_per_week'].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_kpi("Total Students",  f"{total:,}",          icon="🎓", color="#6C63FF")
    with c2: render_kpi("Avg CGPA",        f"{avg_cgpa:.2f}/10",  icon="📈", color="#00D4AA")
    with c3: render_kpi("Pass Rate",       f"{pass_rate:.1f}%",   icon="✅", color="#00C853")
    with c4: render_kpi("Top Performers",  f"{top_performers}",   icon="🏆", color="#FFD740")
    with c5: render_kpi("Avg Attendance",  f"{avg_attendance:.1f}%", icon="📅", color="#40C4FF")
    with c6: render_kpi("Avg Study Hrs",   f"{avg_study:.1f}/wk", icon="📚", color="#FF6B6B")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Grade Distribution ─────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(grade_distribution_pie(df), use_container_width=True)
    with col2:
        st.plotly_chart(grade_distribution_bar(df), use_container_width=True)

    # ── Row 2: CGPA Histogram & Scatter ──────────────────────
    col3, col4 = st.columns([1, 1])
    with col3:
        st.plotly_chart(cgpa_histogram(df), use_container_width=True)
    with col4:
        st.plotly_chart(attendance_vs_cgpa_scatter(df), use_container_width=True)

    # ── Row 3: Department & Gender ────────────────────────────
    col5, col6 = st.columns([1, 1])
    with col5:
        st.plotly_chart(cgpa_by_department(df), use_container_width=True)
    with col6:
        st.plotly_chart(performance_by_gender(df), use_container_width=True)

    # ── Row 4: Study Hours Boxplot ────────────────────────────
    st.plotly_chart(study_hours_boxplot(df), use_container_width=True)

    # ── Summary Stats Table ───────────────────────────────────
    st.markdown("### 📋 Statistical Summary")
    numeric_cols = [
        'prev_cgpa', 'attendance_pct', 'study_hours_per_week',
        'assignment_completion_pct', 'midterm_score', 'final_cgpa'
    ]
    summary = df[numeric_cols].describe().T.round(2)
    summary.index = [
        'Prev CGPA', 'Attendance %', 'Study Hrs/Week',
        'Assignment %', 'Mid-Term Score', 'Final CGPA'
    ]
    st.dataframe(
        summary.style.background_gradient(cmap='viridis', axis=1),
        use_container_width=True
    )

    # ── Department Filter Drill-down ──────────────────────────
    st.markdown("### 🔍 Department-wise Drill-down")
    depts = ['All'] + sorted(df['department'].unique().tolist())
    sel = st.selectbox("Select Department", depts, key="dept_drilldown")
    filtered = df if sel == 'All' else df[df['department'] == sel]

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("Students", len(filtered))
    with dc2:
        st.metric("Avg CGPA", f"{filtered['final_cgpa'].mean():.2f}")
    with dc3:
        pass_r = (filtered['performance'] == 'Pass').sum() / len(filtered) * 100
        st.metric("Pass Rate", f"{pass_r:.1f}%")

    st.plotly_chart(grade_distribution_bar(filtered), use_container_width=True)
