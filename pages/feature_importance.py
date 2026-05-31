"""
Feature Importance Page — Visualizes which factors drive student performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizer import feature_importance_chart, model_comparison_chart, _base_layout, GRID_COLOR, PALETTE
from utils.preprocessor import FEATURE_DISPLAY_NAMES


def show(predictor, df):
    st.markdown("## 🎯 Feature Importance Analysis")
    st.markdown("<p style='color:#aaa;margin-top:-10px'>Understand which factors have the greatest influence on student performance.</p>", unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🏆 Feature Importance", "🤖 Model Comparison", "📊 Model Details"])

    # ── Tab 1: Feature Importance ─────────────────────────────
    with tab1:
        importance = predictor.get_feature_importance()
        if not importance:
            st.warning("No feature importance data found. Please train models first.")
            return

        st.markdown("### 🏆 Top Factors Influencing Student Performance")
        st.info("Feature importances are derived from the **Random Forest** model, which calculates how much each feature reduces uncertainty when making predictions.")

        # Top N Slider
        top_n = st.slider("Show Top N Features", 5, min(20, len(importance)), 15, key="top_n_slider")
        st.plotly_chart(feature_importance_chart(importance, top_n=top_n), use_container_width=True)

        # ── Top 3 Insight Cards ───────────────────────────
        st.markdown("### 💡 Key Insights")
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_items[:3]

        insight_texts = {
            'midterm_score': "Mid-term scores are the most direct signal of exam readiness and correlate strongly with final outcomes.",
            'prev_cgpa': "Past academic performance is a reliable predictor — students with higher CGPAs maintain momentum.",
            'attendance_pct': "Consistent attendance keeps students engaged with course material and reduces knowledge gaps.",
            'study_hours_per_week': "Dedicated study time directly translates to better conceptual understanding and retention.",
            'assignment_completion_pct': "Regular assignment completion reinforces learning and keeps students on track.",
            'num_backlogs': "Backlogs create compounding academic pressure and are a strong negative predictor.",
            'stress_level': "High stress impairs memory, focus, and performance across all academic activities.",
            'participation_score': "Active participation improves comprehension and teacher-student rapport.",
        }
        icons = ['🥇', '🥈', '🥉']

        cols = st.columns(3)
        for i, (feat, imp) in enumerate(top3):
            with cols[i]:
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
                insight = insight_texts.get(feat, f"{display_name} significantly influences student outcomes.")
                pct = imp * 100
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(108,99,255,0.15),rgba(0,212,170,0.08));
                            border:1px solid rgba(108,99,255,0.3);border-radius:16px;padding:20px;
                            min-height:160px">
                  <div style="font-size:28px;margin-bottom:8px">{icons[i]}</div>
                  <p style="font-size:15px;font-weight:700;color:#b8b0ff;margin:0 0 4px">{display_name}</p>
                  <p style="font-size:22px;font-weight:800;color:#00D4AA;margin:0 0 8px">{pct:.2f}%</p>
                  <p style="font-size:12px;color:#aaa;margin:0;line-height:1.5">{insight}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Full Importance Table ─────────────────────────
        st.markdown("### 📋 Full Feature Importance Table")
        importance_df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Feature': FEATURE_DISPLAY_NAMES.get(k, k),
                'Importance Score': round(v, 6),
                'Importance %': f"{v*100:.3f}%",
            }
            for i, (k, v) in enumerate(sorted_items)
        ])
        st.dataframe(
            importance_df.style.background_gradient(
                subset=['Importance Score'], cmap='viridis'
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 2: Model Comparison ───────────────────────────────
    with tab2:
        st.markdown("### 🤖 ML Model Performance Comparison")
        results = predictor.get_model_results()
        if not results:
            st.warning("No model results found. Please train models first.")
            return

        st.plotly_chart(model_comparison_chart(results), use_container_width=True)

        # Radar comparison
        st.markdown("### 🕸️ Model Metrics Radar")
        models = list(results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'cv_mean']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'CV Score']

        fig_radar = go.Figure()
        for i, model in enumerate(models):
            vals = [results[model].get(m, 0) * 100 for m in metrics]
            vals_closed = vals + [vals[0]]
            labels_closed = metric_labels + [metric_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=labels_closed,
                fill='toself', name=model,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                fillcolor=f'rgba({int(PALETTE[i%len(PALETTE)][1:3],16)},{int(PALETTE[i%len(PALETTE)][3:5],16)},{int(PALETTE[i%len(PALETTE)][5:],16)},0.1)',
            ))
        fig_radar.update_layout(**_base_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR),
                bgcolor='rgba(0,0,0,0)',
            ),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
            height=500,
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 3: Model Details ──────────────────────────────────
    with tab3:
        st.markdown("### 📊 Detailed Model Metrics")
        results = predictor.get_model_results()
        if not results:
            st.warning("No model results found.")
            return

        best_model = predictor.get_best_model_name()
        selected_model = st.selectbox("Select Model", list(results.keys()), key="model_detail_select")
        r = results[selected_model]

        is_best = selected_model == best_model
        badge = '&nbsp;🏆 Best Model' if is_best else ''
        st.markdown(f"""
        <div style="background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.3);
                    border-radius:12px;padding:16px;margin-bottom:16px">
          <span style="font-size:18px;font-weight:700;color:#b8b0ff">{selected_model}{badge}</span>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",   f"{r['accuracy']*100:.2f}%")
        m2.metric("F1 Score",   f"{r['f1_score']*100:.2f}%")
        m3.metric("Precision",  f"{r['precision']*100:.2f}%")
        m4.metric("Recall",     f"{r['recall']*100:.2f}%")
        m5.metric("CV Score",   f"{r['cv_mean']*100:.2f}% ± {r['cv_std']*100:.2f}%")

        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = r['confusion_matrix']
        grade_classes = predictor.get_grade_classes()

        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=grade_classes, y=grade_classes,
            colorscale='Blues',
            text=cm, texttemplate='%{text}', textfont=dict(size=13),
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        ))
        fig_cm.update_layout(**_base_layout(
            title=f'Confusion Matrix — {selected_model}',
            xaxis=dict(title='Predicted Grade'),
            yaxis=dict(title='Actual Grade'),
            height=380,
        ))
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class metrics
        st.markdown("#### Per-Class Classification Report")
        report = r['classification_report']
        report_rows = []
        for cls in grade_classes:
            if cls in report:
                row = report[cls]
                report_rows.append({
                    'Grade': cls,
                    'Precision': f"{row['precision']*100:.1f}%",
                    'Recall': f"{row['recall']*100:.1f}%",
                    'F1-Score': f"{row['f1-score']*100:.1f}%",
                    'Support': int(row['support']),
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
