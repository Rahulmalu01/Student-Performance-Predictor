"""
Prediction Page — Interactive form for student performance prediction.
"""

import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizer import cgpa_gauge, grade_probability_bar, student_vs_average_radar
from utils.recommender import generate_recommendations, get_strength_areas


GRADE_EMOJI = {'A+': '🏆', 'A': '⭐', 'B': '👍', 'C': '📘', 'D': '⚠️', 'F': '❌'}
GRADE_BG = {
    'A+': 'linear-gradient(135deg,#00C853,#00897B)',
    'A':  'linear-gradient(135deg,#43A047,#00ACC1)',
    'B':  'linear-gradient(135deg,#1E88E5,#5E35B1)',
    'C':  'linear-gradient(135deg,#FB8C00,#F4511E)',
    'D':  'linear-gradient(135deg,#E53935,#C62828)',
    'F':  'linear-gradient(135deg,#B71C1C,#880E4F)',
}


def _section(title):
    st.markdown(f"""
    <div style="background:rgba(108,99,255,0.08);border-left:3px solid #6C63FF;
                padding:8px 14px;border-radius:0 8px 8px 0;margin:18px 0 10px">
      <span style="font-weight:600;color:#b8b0ff;font-size:14px">{title}</span>
    </div>""", unsafe_allow_html=True)


def show(predictor, df):
    st.markdown("## 🔮 Student Performance Predictor")
    st.markdown("<p style='color:#aaa;margin-top:-10px'>Fill in the student details below to get an instant AI-powered grade prediction.</p>", unsafe_allow_html=True)
    st.markdown("---")

    df_means = df.mean(numeric_only=True).to_dict()

    with st.form("prediction_form"):
        # ── Academic Information ──────────────────────────
        _section("🎓 Academic Information")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            prev_cgpa = st.slider("Previous CGPA", 0.0, 10.0, 7.0, 0.1,
                                  help="CGPA from last semester (0–10 scale)")
        with ac2:
            midterm_score = st.slider("Mid-Term Score", 0.0, 100.0, 65.0, 0.5,
                                      help="Score in mid-term examination (out of 100)")
        with ac3:
            num_backlogs = st.number_input("Number of Backlogs", 0, 10, 0, 1,
                                           help="Pending failed subjects")

        ac4, ac5, ac6 = st.columns(3)
        with ac4:
            attendance_pct = st.slider("Attendance %", 0.0, 100.0, 75.0, 0.5)
        with ac5:
            assignment_completion_pct = st.slider("Assignment Completion %", 0.0, 100.0, 80.0, 0.5)
        with ac6:
            semester = st.selectbox("Current Semester", list(range(1, 9)), index=3)

        # ── Study & Engagement ────────────────────────────
        _section("📚 Study & Engagement")
        se1, se2, se3 = st.columns(3)
        with se1:
            study_hours_per_week = st.slider("Study Hours / Week", 0.0, 30.0, 10.0, 0.5)
        with se2:
            participation_score = st.slider("Class Participation (1–10)", 1.0, 10.0, 5.0, 0.5)
        with se3:
            library_hours_per_week = st.slider("Library Hours / Week", 0.0, 20.0, 3.0, 0.5)
        se4, se5 = st.columns(2)
        with se4:
            extracurricular_activities = st.number_input("Extracurricular Activities", 0, 5, 1, 1)
        with se5:
            teacher_interaction = st.selectbox("Teacher Interaction Frequency",
                                               options=[0, 1, 2],
                                               format_func=lambda x: ['Never', 'Sometimes', 'Often'][x],
                                               index=1)

        # ── Lifestyle & Wellbeing ─────────────────────────
        _section("🧘 Lifestyle & Wellbeing")
        lw1, lw2 = st.columns(2)
        with lw1:
            sleep_hours_per_night = st.slider("Sleep Hours / Night", 4.0, 10.0, 7.0, 0.5)
        with lw2:
            stress_level = st.slider("Stress Level (1–10)", 1.0, 10.0, 5.0, 0.5,
                                     help="1 = very low stress, 10 = extremely stressed")

        # ── Personal & Socioeconomic ──────────────────────
        _section("👤 Personal & Socioeconomic")
        ps1, ps2, ps3 = st.columns(3)
        with ps1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
        with ps2:
            department = st.selectbox("Department",
                                      ['Computer Science', 'Electronics', 'Mechanical', 'Civil', 'Business'])
        with ps3:
            age = st.number_input("Age", 17, 30, 20, 1)

        ps4, ps5, ps6 = st.columns(3)
        with ps4:
            income_bracket = st.selectbox("Family Income Bracket",
                                          ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
                                          index=2)
        with ps5:
            internet_access = st.selectbox("Internet Access", [1, 0],
                                           format_func=lambda x: 'Yes' if x else 'No')
        with ps6:
            part_time_job = st.selectbox("Part-Time Job", [0, 1],
                                         format_func=lambda x: 'Yes' if x else 'No')

        first_gen_student = st.checkbox("First Generation College Student", value=False)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True,
                                          type="primary")

    if submitted:
        feature_dict = {
            'age': age, 'gender': gender, 'department': department,
            'semester': semester, 'income_bracket': income_bracket,
            'internet_access': internet_access, 'part_time_job': part_time_job,
            'first_gen_student': int(first_gen_student),
            'prev_cgpa': prev_cgpa, 'attendance_pct': attendance_pct,
            'study_hours_per_week': study_hours_per_week,
            'assignment_completion_pct': assignment_completion_pct,
            'midterm_score': midterm_score, 'num_backlogs': num_backlogs,
            'participation_score': participation_score,
            'library_hours_per_week': library_hours_per_week,
            'extracurricular_activities': extracurricular_activities,
            'sleep_hours_per_night': sleep_hours_per_night,
            'stress_level': stress_level,
            'teacher_interaction': teacher_interaction,
        }

        result = predictor.predict(feature_dict)
        grade = result['grade']

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        # ── Grade Result Card ─────────────────────────────
        emoji = GRADE_EMOJI.get(grade, '📊')
        bg = GRADE_BG.get(grade, 'linear-gradient(135deg,#6C63FF,#00D4AA)')
        st.markdown(f"""
        <div style="background:{bg};border-radius:20px;padding:30px;text-align:center;
                    margin:10px 0 20px;box-shadow:0 8px 32px rgba(0,0,0,0.4)">
          <div style="font-size:52px;margin-bottom:8px">{emoji}</div>
          <h1 style="font-size:56px;font-weight:800;color:white;margin:0;letter-spacing:-1px">
            Grade {grade}
          </h1>
          <p style="color:rgba(255,255,255,0.85);font-size:16px;margin:8px 0 0">
            {result['description']}
          </p>
          <p style="color:rgba(255,255,255,0.65);font-size:13px;margin:6px 0 0">
            Confidence: <strong>{result['confidence']:.1f}%</strong> &nbsp;|&nbsp;
            Model: <strong>{result['model_used']}</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge + Probability ───────────────────────────
        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(cgpa_gauge(result['cgpa_estimate']), use_container_width=True)
        with g2:
            st.plotly_chart(grade_probability_bar(result['probabilities']), use_container_width=True)

        # ── Radar Chart ───────────────────────────────────
        st.plotly_chart(
            student_vs_average_radar(feature_dict, df_means),
            use_container_width=True
        )

        # ── Strengths ─────────────────────────────────────
        strengths = get_strength_areas(feature_dict)
        if strengths:
            st.markdown("### 💪 Your Strengths")
            scols = st.columns(min(len(strengths), 4))
            for i, s in enumerate(strengths):
                with scols[i % 4]:
                    st.markdown(f"""
                    <div style="background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.3);
                                border-radius:12px;padding:12px;text-align:center">
                      <div style="font-size:24px">{s['icon']}</div>
                      <p style="font-size:12px;font-weight:600;color:#69F0AE;margin:4px 0 2px">{s['area']}</p>
                      <p style="font-size:11px;color:#aaa;margin:0">{s['value']}</p>
                    </div>""", unsafe_allow_html=True)

        # ── Quick Recommendations Preview ─────────────────
        recs = generate_recommendations(feature_dict)
        if recs:
            st.markdown("### ⚡ Top Improvement Areas")
            for rec in recs[:3]:
                priority_color = {'Critical': '#FF1744', 'High': '#FF6D00', 'Medium': '#7C4DFF', 'Low': '#00BCD4'}
                pc = priority_color.get(rec['priority'], '#888')
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                            border-left:4px solid {rec['color']};border-radius:12px;padding:14px 16px;
                            margin:6px 0">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:600;font-size:15px">{rec['title']}</span>
                    <span style="background:{pc};color:white;font-size:11px;font-weight:600;
                                 padding:2px 10px;border-radius:20px">{rec['priority']}</span>
                  </div>
                  <p style="font-size:13px;color:#aaa;margin:6px 0 0">{rec['impact']}</p>
                </div>""", unsafe_allow_html=True)
            st.info("💡 Go to the **Recommendations** page for the full personalized action plan.")

        # ── Store in session for Recommendations page ──────
        st.session_state['last_prediction'] = {
            'feature_dict': feature_dict,
            'result': result,
        }
