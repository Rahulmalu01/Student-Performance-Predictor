"""
Analytics Page — Comparative analytics between students and groups.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizer import correlation_heatmap, cgpa_by_department, CHART_TEMPLATE, PALETTE, GRID_COLOR, _base_layout


def show(df: pd.DataFrame):
    st.markdown("## 📈 Comparative Analytics")
    st.markdown("<p style='color:#aaa;margin-top:-10px'>Explore performance patterns across different student groups and features.</p>", unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Group Comparisons", "🔗 Correlation Analysis",
        "📉 Feature vs Grade", "🏫 Department Deep-Dive"
    ])

    # ── Tab 1: Group Comparisons ──────────────────────────────
    with tab1:
        st.markdown("### 📊 CGPA Comparison Across Groups")

        col1, col2 = st.columns(2)
        with col1:
            # Income vs CGPA
            income_order = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
            inc_df = df.groupby('income_bracket')['final_cgpa'].mean().reindex(income_order).reset_index()
            inc_df.columns = ['Income', 'Avg CGPA']
            fig_income = go.Figure(go.Bar(
                x=inc_df['Income'], y=inc_df['Avg CGPA'],
                marker=dict(color=PALETTE, line=dict(color='rgba(255,255,255,0.05)', width=1)),
                text=inc_df['Avg CGPA'].round(2), textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg CGPA: %{y:.2f}<extra></extra>',
            ))
            fig_income.update_layout(**_base_layout(
                title='Avg CGPA by Income Bracket',
                xaxis=dict(title='Income', gridcolor=GRID_COLOR),
                yaxis=dict(title='Avg CGPA', range=[0, 11], gridcolor=GRID_COLOR),
            ))
            st.plotly_chart(fig_income, use_container_width=True)

        with col2:
            # Internet vs No Internet
            net_df = df.groupby('internet_access')['final_cgpa'].agg(['mean', 'std']).reset_index()
            net_df['label'] = net_df['internet_access'].map({0: 'No Internet', 1: 'Has Internet'})
            fig_net = go.Figure(go.Bar(
                x=net_df['label'], y=net_df['mean'],
                error_y=dict(type='data', array=net_df['std'], visible=True),
                marker=dict(color=['#FF6B6B', '#00D4AA']),
                text=net_df['mean'].round(2), textposition='outside',
            ))
            fig_net.update_layout(**_base_layout(
                title='CGPA: Internet Access Impact',
                xaxis=dict(title='', gridcolor=GRID_COLOR),
                yaxis=dict(title='Avg CGPA', range=[0, 11], gridcolor=GRID_COLOR),
            ))
            st.plotly_chart(fig_net, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            # Part-time job impact
            job_df = df.groupby('part_time_job')['final_cgpa'].agg(['mean', 'std']).reset_index()
            job_df['label'] = job_df['part_time_job'].map({0: 'No Job', 1: 'Part-Time Job'})
            fig_job = go.Figure(go.Bar(
                x=job_df['label'], y=job_df['mean'],
                error_y=dict(type='data', array=job_df['std'], visible=True),
                marker=dict(color=['#6C63FF', '#FF6D00']),
                text=job_df['mean'].round(2), textposition='outside',
            ))
            fig_job.update_layout(**_base_layout(
                title='CGPA: Part-Time Job Impact',
                xaxis=dict(title='', gridcolor=GRID_COLOR),
                yaxis=dict(title='Avg CGPA', range=[0, 11], gridcolor=GRID_COLOR),
            ))
            st.plotly_chart(fig_job, use_container_width=True)

        with col4:
            # First gen vs not
            fg_df = df.groupby('first_gen_student')['final_cgpa'].agg(['mean', 'std']).reset_index()
            fg_df['label'] = fg_df['first_gen_student'].map({0: 'Not First-Gen', 1: 'First-Gen Student'})
            fig_fg = go.Figure(go.Bar(
                x=fg_df['label'], y=fg_df['mean'],
                error_y=dict(type='data', array=fg_df['std'], visible=True),
                marker=dict(color=['#40C4FF', '#FFD740']),
                text=fg_df['mean'].round(2), textposition='outside',
            ))
            fig_fg.update_layout(**_base_layout(
                title='CGPA: First-Generation Students',
                xaxis=dict(title='', gridcolor=GRID_COLOR),
                yaxis=dict(title='Avg CGPA', range=[0, 11], gridcolor=GRID_COLOR),
            ))
            st.plotly_chart(fig_fg, use_container_width=True)

        # Backlogs distribution
        st.markdown("### 📦 Backlog Distribution by Grade")
        grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
        backlog_df = df.groupby('final_grade')['num_backlogs'].mean().reindex(grade_order).reset_index()
        colors = ['#00C853','#69F0AE','#40C4FF','#FFD740','#FF6D00','#FF1744']
        fig_bl = go.Figure(go.Bar(
            x=backlog_df['final_grade'], y=backlog_df['num_backlogs'],
            marker=dict(color=colors),
            text=backlog_df['num_backlogs'].round(2), textposition='outside',
        ))
        fig_bl.update_layout(**_base_layout(
            title='Average Backlogs per Grade Category',
            xaxis=dict(title='Grade', gridcolor=GRID_COLOR),
            yaxis=dict(title='Avg Backlogs', gridcolor=GRID_COLOR),
        ))
        st.plotly_chart(fig_bl, use_container_width=True)

    # ── Tab 2: Correlation Analysis ───────────────────────────
    with tab2:
        st.markdown("### 🔗 Feature Correlation Matrix")
        st.info("This heatmap shows Pearson correlation coefficients between all numeric features. Values close to 1 or -1 indicate strong relationships.")
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

        # Top correlations with final CGPA
        st.markdown("### 🎯 Top Correlations with Final CGPA")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if 'final_cgpa' in numeric_cols:
            corrs = df[numeric_cols].corr()['final_cgpa'].drop('final_cgpa').sort_values(key=abs, ascending=False)
            corr_df = pd.DataFrame({'Feature': corrs.index, 'Correlation': corrs.values})
            colors_corr = ['#00C853' if v > 0 else '#FF1744' for v in corr_df['Correlation']]
            fig_corr = go.Figure(go.Bar(
                x=corr_df['Correlation'], y=corr_df['Feature'],
                orientation='h',
                marker=dict(color=colors_corr),
                text=corr_df['Correlation'].round(3), textposition='outside',
            ))
            fig_corr.update_layout(**_base_layout(
                title='Feature Correlations with Final CGPA',
                xaxis=dict(title='Pearson Correlation', gridcolor=GRID_COLOR, range=[-1, 1]),
                yaxis=dict(automargin=True, gridcolor=GRID_COLOR),
                height=500,
            ))
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 3: Feature vs Grade Scatter ───────────────────────
    with tab3:
        st.markdown("### 📉 Feature vs Final CGPA")
        numeric_features = [
            'prev_cgpa', 'attendance_pct', 'study_hours_per_week',
            'assignment_completion_pct', 'midterm_score', 'participation_score',
            'sleep_hours_per_night', 'stress_level', 'library_hours_per_week'
        ]
        feat_labels = {
            'prev_cgpa': 'Previous CGPA',
            'attendance_pct': 'Attendance %',
            'study_hours_per_week': 'Study Hours/Week',
            'assignment_completion_pct': 'Assignment Completion %',
            'midterm_score': 'Mid-Term Score',
            'participation_score': 'Participation Score',
            'sleep_hours_per_night': 'Sleep Hours/Night',
            'stress_level': 'Stress Level',
            'library_hours_per_week': 'Library Hours/Week',
        }

        selected_feat = st.selectbox(
            "Select Feature to Analyze",
            options=numeric_features,
            format_func=lambda x: feat_labels.get(x, x)
        )

        sample_df = df.sample(min(500, len(df)), random_state=42)
        grade_colors_map = {'A+':'#00C853','A':'#69F0AE','B':'#40C4FF','C':'#FFD740','D':'#FF6D00','F':'#FF1744'}

        fig_scatter = px.scatter(
            sample_df, x=selected_feat, y='final_cgpa',
            color='final_grade',
            color_discrete_map=grade_colors_map,
            trendline='ols',
            labels={selected_feat: feat_labels.get(selected_feat, selected_feat), 'final_cgpa': 'Final CGPA'},
            template=CHART_TEMPLATE,
            hover_data=['department', 'semester'],
        )
        fig_scatter.update_layout(**_base_layout(
            title=f'{feat_labels.get(selected_feat)} vs Final CGPA',
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR),
            legend=dict(orientation='h', y=-0.2),
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Distribution of selected feature by grade
        fig_box = go.Figure()
        for g, c in grade_colors_map.items():
            subset = df[df['final_grade'] == g][selected_feat]
            if len(subset) > 0:
                fig_box.add_trace(go.Box(y=subset, name=g, marker_color=c, boxmean=True))
        fig_box.update_layout(**_base_layout(
            title=f'{feat_labels.get(selected_feat)} Distribution by Grade',
            xaxis=dict(title='Grade', gridcolor=GRID_COLOR),
            yaxis=dict(title=feat_labels.get(selected_feat, selected_feat), gridcolor=GRID_COLOR),
            showlegend=False,
        ))
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Tab 4: Department Deep-Dive ───────────────────────────
    with tab4:
        st.markdown("### 🏫 Department Performance Analysis")
        st.plotly_chart(cgpa_by_department(df), use_container_width=True)

        # Semester progression by dept
        st.markdown("### 📈 CGPA Progression by Semester")
        dept_list = sorted(df['department'].unique().tolist())
        selected_depts = st.multiselect("Select Departments", dept_list, default=dept_list[:3])

        if selected_depts:
            sem_df = df[df['department'].isin(selected_depts)].groupby(
                ['department', 'semester'])['final_cgpa'].mean().reset_index()
            fig_sem = px.line(
                sem_df, x='semester', y='final_cgpa', color='department',
                markers=True,
                template=CHART_TEMPLATE,
                color_discrete_sequence=PALETTE,
                labels={'semester': 'Semester', 'final_cgpa': 'Avg CGPA', 'department': 'Department'},
            )
            fig_sem.update_layout(**_base_layout(
                title='Average CGPA Progression by Semester',
                xaxis=dict(title='Semester', gridcolor=GRID_COLOR, dtick=1),
                yaxis=dict(title='Average CGPA', gridcolor=GRID_COLOR),
                legend=dict(orientation='h', y=-0.2),
            ))
            st.plotly_chart(fig_sem, use_container_width=True)

        # Stress level comparison
        st.markdown("### 🧘 Stress Level by Department")
        stress_dept = df.groupby('department')['stress_level'].mean().reset_index()
        fig_stress = px.bar(
            stress_dept, x='department', y='stress_level',
            color='stress_level', color_continuous_scale='RdYlGn_r',
            template=CHART_TEMPLATE,
            text=stress_dept['stress_level'].round(2),
        )
        fig_stress.update_traces(textposition='outside')
        fig_stress.update_layout(**_base_layout(
            title='Average Stress Level by Department',
            xaxis=dict(title='Department', gridcolor=GRID_COLOR),
            yaxis=dict(title='Avg Stress Level (1–10)', gridcolor=GRID_COLOR, range=[0, 12]),
            coloraxis_showscale=False,
        ))
        st.plotly_chart(fig_stress, use_container_width=True)
