"""
Recommendations Page — Full personalized academic improvement plan.
"""

import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.recommender import generate_recommendations, get_strength_areas, PRIORITY_ORDER


PRIORITY_COLORS = {
    'Critical': '#FF1744',
    'High':     '#FF6D00',
    'Medium':   '#7C4DFF',
    'Low':      '#00BCD4',
}

PRIORITY_BG = {
    'Critical': 'rgba(255,23,68,0.12)',
    'High':     'rgba(255,109,0,0.12)',
    'Medium':   'rgba(124,77,255,0.12)',
    'Low':      'rgba(0,188,212,0.12)',
}


def _render_rec_card(rec):
    pc = PRIORITY_COLORS.get(rec['priority'], '#888')
    pbg = PRIORITY_BG.get(rec['priority'], 'rgba(255,255,255,0.05)')

    st.markdown(f"""
    <div style="background:{pbg};border:1px solid {pc}33;border-left:4px solid {pc};
                border-radius:14px;padding:20px 22px;margin:10px 0">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
        <div>
          <h4 style="color:white;margin:0 0 4px;font-size:16px">{rec['title']}</h4>
          <p style="color:#aaa;font-size:12px;margin:0">
            Current: <strong style="color:{pc}">{rec['current_value']:.1f}</strong>
            &nbsp;→&nbsp; Target: <strong style="color:#00D4AA">{rec['target_value']}</strong>
          </p>
        </div>
        <span style="background:{pc};color:white;font-size:11px;font-weight:700;
                     padding:3px 12px;border-radius:20px;white-space:nowrap">{rec['priority']}</span>
      </div>
      <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 14px;margin:12px 0 8px">
        <p style="color:#ffd740;font-size:12px;font-weight:600;margin:0 0 4px">📊 Impact</p>
        <p style="color:#ccc;font-size:13px;margin:0;line-height:1.5">{rec['impact']}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"💡 Action Tips & Resources for {rec['title'].split(' ', 1)[-1]}"):
        st.markdown("**Recommended Actions:**")
        for tip in rec['tips']:
            st.markdown(f"- {tip}")
        if rec.get('resources'):
            st.markdown("**📚 Helpful Resources:**")
            for res in rec['resources']:
                st.markdown(f"- [{res['name']}]({res['url']})")


def show(predictor):
    st.markdown("## 💡 Personalized Recommendations")
    st.markdown("<p style='color:#aaa;margin-top:-10px'>AI-powered, priority-ranked improvement plan based on student profile.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Check if prediction was made ─────────────────────────
    if 'last_prediction' not in st.session_state:
        st.markdown("""
        <div style="background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.3);
                    border-radius:16px;padding:40px;text-align:center">
          <div style="font-size:48px;margin-bottom:12px">🎯</div>
          <h3 style="color:#b8b0ff;margin:0 0 8px">No Prediction Yet</h3>
          <p style="color:#aaa;margin:0">Go to the <strong>Predict Performance</strong> page,
          fill in a student profile, and come back here to see personalized recommendations.</p>
        </div>""", unsafe_allow_html=True)
        return

    data = st.session_state['last_prediction']
    feature_dict = data['feature_dict']
    result = data['result']

    # ── Summary Banner ────────────────────────────────────────
    grade = result['grade']
    grade_colors = {'A+':'#00C853','A':'#43A047','B':'#1E88E5','C':'#FB8C00','D':'#E53935','F':'#B71C1C'}
    gc = grade_colors.get(grade, '#6C63FF')

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(108,99,255,0.2),rgba(0,212,170,0.1));
                border:1px solid rgba(108,99,255,0.3);border-radius:16px;padding:18px 22px;
                display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:16px">
      <div style="text-align:center;min-width:80px">
        <p style="font-size:36px;font-weight:800;color:{gc};margin:0">Grade {grade}</p>
        <p style="font-size:12px;color:#aaa;margin:0">Predicted Grade</p>
      </div>
      <div style="flex:1;min-width:200px">
        <p style="font-size:14px;color:#ddd;margin:0 0 4px">
          <strong>Confidence:</strong> {result['confidence']:.1f}% &nbsp;|&nbsp;
          <strong>Est. CGPA:</strong> {result['cgpa_estimate']:.1f}/10
        </p>
        <p style="font-size:13px;color:#aaa;margin:0">{result['description']}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Strengths ─────────────────────────────────────────────
    strengths = get_strength_areas(feature_dict)
    if strengths:
        st.markdown("### 💪 Current Strengths")
        s_cols = st.columns(min(len(strengths), 4))
        for i, s in enumerate(strengths):
            with s_cols[i % 4]:
                st.markdown(f"""
                <div style="background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.25);
                            border-radius:12px;padding:14px;text-align:center;margin-bottom:8px">
                  <div style="font-size:22px">{s['icon']}</div>
                  <p style="font-size:12px;font-weight:600;color:#69F0AE;margin:4px 0 2px">{s['area']}</p>
                  <p style="font-size:11px;color:#aaa;margin:0">{s['value']}</p>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────
    recs = generate_recommendations(feature_dict)

    if not recs:
        st.markdown("""
        <div style="background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.3);
                    border-radius:16px;padding:30px;text-align:center">
          <div style="font-size:48px;margin-bottom:8px">🎉</div>
          <h3 style="color:#00C853;margin:0 0 8px">Outstanding Profile!</h3>
          <p style="color:#aaa;margin:0">This student is performing excellently across all measured areas.
          Keep up the great work!</p>
        </div>""", unsafe_allow_html=True)
        return

    # Priority summary
    from collections import Counter
    priority_counts = Counter(r['priority'] for r in recs)
    p_cols = st.columns(4)
    for i, (pr, label) in enumerate(zip(
        ['Critical', 'High', 'Medium', 'Low'],
        ['🚨 Critical', '🔴 High', '🟣 Medium', '🔵 Low']
    )):
        cnt = priority_counts.get(pr, 0)
        with p_cols[i]:
            color = PRIORITY_COLORS.get(pr, '#888')
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid {color}44;
                        border-radius:10px;padding:12px;text-align:center">
              <p style="font-size:24px;font-weight:700;color:{color};margin:0">{cnt}</p>
              <p style="font-size:11px;color:#aaa;margin:0">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter by priority
    all_priorities = ['All'] + [p for p in ['Critical', 'High', 'Medium', 'Low']
                                if p in priority_counts]
    selected_priority = st.selectbox("Filter by Priority", all_priorities, key="rec_priority_filter")

    filtered_recs = recs if selected_priority == 'All' else [r for r in recs if r['priority'] == selected_priority]

    st.markdown(f"### 📋 Improvement Plan ({len(filtered_recs)} area{'s' if len(filtered_recs) != 1 else ''})")

    for rec in filtered_recs:
        _render_rec_card(rec)

    # ── Study Plan Summary ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Suggested Weekly Study Plan")

    study_plan = []
    features = {r['key'] for r in recs}

    if 'attendance' in features:
        study_plan.append(("Monday & Wednesday", "Focus on attending ALL scheduled classes. Prepare the evening before."))
    if 'study_hours' in features or 'assignments' in features:
        study_plan.append(("Tuesday & Thursday", "Dedicated 3-hour study sessions using the Pomodoro Technique (25+5 min cycles)."))
    if 'midterm' in features:
        study_plan.append(("Friday", "Practice past exam papers and active recall testing on the week's material."))
    if 'stress' in features or 'sleep' in features:
        study_plan.append(("Saturday", "Rest and recovery day. Light revision only. Aim for 8+ hours of sleep."))
    study_plan.append(("Sunday", "Weekly review: re-read notes, update assignment tracker, plan next week."))

    for day, plan in study_plan:
        st.markdown(f"""
        <div style="display:flex;gap:16px;align-items:flex-start;padding:10px 0;
                    border-bottom:1px solid rgba(255,255,255,0.06)">
          <span style="background:rgba(108,99,255,0.2);color:#b8b0ff;border-radius:8px;
                       padding:4px 12px;font-size:12px;font-weight:600;white-space:nowrap;min-width:170px">
            {day}</span>
          <span style="color:#ddd;font-size:13px;line-height:1.5">{plan}</span>
        </div>""", unsafe_allow_html=True)
