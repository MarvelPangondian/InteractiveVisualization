import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Set page configuration
st.set_page_config(
    page_title="Student Life Balance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate dummy data
@st.cache_data
def generate_data(n_samples=100):
    np.random.seed(42)
    
    # Generate realistic and slightly correlated data
    study_hours = np.clip(np.random.normal(6, 1.5, n_samples), 1, 12)
    sleep_hours = np.clip(np.random.normal(7, 1, n_samples), 4, 10)
    
    # Less study typically means more social/extracurricular
    extracurricular_hours = np.clip(np.random.normal(2, 1, n_samples) + (8 - study_hours) * 0.2, 0, 6)
    social_hours = np.clip(np.random.normal(3, 1.2, n_samples) + (8 - study_hours) * 0.3, 0, 8)
    
    physical_hours = np.clip(np.random.normal(1.5, 0.8, n_samples), 0, 4)
    
    # GPA has positive correlation with study hours and sleep hours
    gpa_base = 2.5 + study_hours * 0.15 + sleep_hours * 0.05 - 0.1 * (np.random.normal(0, 0.5, n_samples))
    gpa = np.clip(gpa_base, 0, 4.0).round(2)
    
    # Stress level is related to study hours and inversely to sleep
    stress_prob = (study_hours / 12) * 0.7 + (1 - (sleep_hours / 10)) * 0.3 + np.random.normal(0, 0.15, n_samples)
    stress_level = []
    for prob in stress_prob:
        if prob < 0.4:
            stress_level.append("Low")
        elif prob < 0.7:
            stress_level.append("Moderate")
        else:
            stress_level.append("High")
    
    # Work-life balance is a function of all factors with added random variation to ensure diversity
    balance_score = (
        (1 - (study_hours / 12) * 0.3) +  # Less study time improves balance, but only slightly
        (sleep_hours / 10) * 0.3 +        # More sleep improves balance
        (extracurricular_hours / 6) * 0.15 + # Some extracurricular activities improve balance
        (social_hours / 8) * 0.15 +       # Social time improves balance
        (physical_hours / 4) * 0.1 +      # Physical activity improves balance
        np.random.normal(0, 0.05, n_samples)  # Add random variation to ensure range of values
    ) / 1.0
    balance_score = np.clip(balance_score, 0, 1).round(2)
    
    # Make absolutely sure we have at least two different balance scores
    if len(np.unique(balance_score)) == 1:
        # If all scores are the same, modify at least one
        balance_score[0] = min(1.0, balance_score[0] + 0.1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Study_Hours_Per_Day': study_hours.round(1),
        'Sleep_Hours_Per_Day': sleep_hours.round(1),
        'Extracurricular_Hours_Per_Day': extracurricular_hours.round(1),
        'Social_Hours_Per_Day': social_hours.round(1),
        'Physical_Activity_Hours_Per_Day': physical_hours.round(1),
        'GPA': gpa,
        'Stress_Level': stress_level,
        'Work_Life_Balance_Score': balance_score
    })
    
    # Add student ID for reference
    df['Student_ID'] = [f'S{i+1:03d}' for i in range(n_samples)]
    
    return df

# Generate the data
df = generate_data(150)

# Sidebar for filtering
st.sidebar.title("Student Life Balance Dashboard")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3771/3771417.png", width=100)

# Add number of students slider
num_students = st.sidebar.slider("Number of Students to Display", 
                              min_value=10, 
                              max_value=len(df), 
                              value=50,
                              step=5)

# Filtering options
st.sidebar.subheader("Filters")

# Stress level filter (multiselect)
stress_options = ['Low', 'Moderate', 'High']
selected_stress = st.sidebar.multiselect(
    "Stress Level",
    options=stress_options,
    default=stress_options
)

# GPA range filter (slider)
min_gpa_value = float(df['GPA'].min())
max_gpa_value = float(df['GPA'].max())

# Ensure min and max values are different
if min_gpa_value == max_gpa_value:
    # If they're the same, create a small range around that value
    min_gpa_value = max(0.0, min_gpa_value - 0.1)
    max_gpa_value = min(4.0, max_gpa_value + 0.1)

# Ensure values are the same type (float)
min_gpa, max_gpa = st.sidebar.slider(
    "GPA Range",
    min_value=float(min_gpa_value),
    max_value=float(max_gpa_value),
    value=(float(min_gpa_value), float(max_gpa_value)),
    step=0.1
)

# Balance score filter
min_balance_value = float(df['Work_Life_Balance_Score'].min())
max_balance_value = float(df['Work_Life_Balance_Score'].max())

# Ensure min and max values are different
if min_balance_value == max_balance_value:
    # If they're the same, create a small range around that value
    min_balance_value = max(0.0, min_balance_value - 0.05)
    max_balance_value = min(1.0, max_balance_value + 0.05)

# Ensure values are the same type (float)
min_balance, max_balance = st.sidebar.slider(
    "Work-Life Balance Score",
    min_value=float(min_balance_value),
    max_value=float(max_balance_value),
    value=(float(min_balance_value), float(max_balance_value)),
    step=0.05
)

# Apply filters
filtered_df = df[
    (df['Stress_Level'].isin(selected_stress)) &
    (df['GPA'] >= min_gpa) & (df['GPA'] <= max_gpa) &
    (df['Work_Life_Balance_Score'] >= min_balance) & (df['Work_Life_Balance_Score'] <= max_balance)
].head(num_students)

# Main dashboard
st.title("Student Life Balance Analytics Dashboard")
st.write(f"Displaying data for {len(filtered_df)} students")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average GPA", f"{filtered_df['GPA'].mean():.2f}")
with col2:
    st.metric("Average Study Hours", f"{filtered_df['Study_Hours_Per_Day'].mean():.1f}")
with col3:
    st.metric("Average Sleep Hours", f"{filtered_df['Sleep_Hours_Per_Day'].mean():.1f}")
with col4:
    st.metric("Average Balance Score", f"{filtered_df['Work_Life_Balance_Score'].mean():.2f}")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Correlation Analysis", "Time Distribution", "Individual Students"])

with tab1:
    st.subheader("Student Life Balance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GPA Distribution by Stress Level
        fig = px.box(filtered_df, x="Stress_Level", y="GPA", 
                    color="Stress_Level", 
                    title="GPA Distribution by Stress Level",
                    color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"})
        fig.update_layout(height=400, boxmode='group', xaxis_title="Stress Level", yaxis_title="GPA")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Work-Life Balance by Stress Level
        fig = px.violin(filtered_df, x="Stress_Level", y="Work_Life_Balance_Score", 
                        color="Stress_Level", box=True,
                        title="Work-Life Balance Distribution by Stress Level",
                        color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Stress Level Distribution
    stress_counts = filtered_df['Stress_Level'].value_counts().reset_index()
    stress_counts.columns = ['Stress_Level', 'Count']
    
    fig = px.pie(stress_counts, values='Count', names='Stress_Level', 
                title='Stress Level Distribution',
                color='Stress_Level',
                color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Correlation Analysis")
    
    # Dropdown for selecting x and y variables
    col1, col2 = st.columns(2)
    
    numeric_columns = [col for col in filtered_df.columns if col not in ['Student_ID', 'Stress_Level']]
    
    with col1:
        x_var = st.selectbox(
            "Select X-axis Variable",
            options=numeric_columns,
            index=0
        )
    
    with col2:
        y_var = st.selectbox(
            "Select Y-axis Variable",
            options=numeric_columns,
            index=5  # Default to GPA
        )
    
    # Scatter plot with zoom capability (enabled by default in Plotly)
    fig = px.scatter(filtered_df, x=x_var, y=y_var, 
                    color="Stress_Level",
                    size="Work_Life_Balance_Score",
                    hover_name="Student_ID",
                    title=f"Relationship between {x_var.replace('_', ' ')} and {y_var.replace('_', ' ')}",
                    color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"},
                    size_max=15)
    
    fig.update_layout(height=600)
    # Enable zoom by default in Plotly
    fig.update_layout(dragmode='zoom')
    
    # Add trendline
    add_trendline = st.checkbox("Show Trendline", value=True)
    if add_trendline:
        fig.update_layout(showlegend=True)
        # Add trendline for each stress level
        for stress in filtered_df['Stress_Level'].unique():
            stress_df = filtered_df[filtered_df['Stress_Level'] == stress]
            fig.add_trace(
                go.Scatter(
                    x=stress_df[x_var],
                    y=stress_df[y_var],
                    mode='lines',
                    name=f'Trendline - {stress}',
                    line=dict(width=2, dash='dash'),
                    showlegend=True
                )
            )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr = filtered_df.drop(['Student_ID', 'Stress_Level'], axis=1).corr()
    
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title="Correlation Between Variables")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Daily Time Distribution")
    
    # Create a time allocation bar chart
    time_cols = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']
    
    # Sample selector
    sample_size = st.slider("Number of Students to Display", min_value=5, max_value=min(30, len(filtered_df)), value=10, step=1)
    
    # Sort options
    sort_by = st.selectbox(
        "Sort Students By:",
        options=['Student_ID', 'GPA', 'Work_Life_Balance_Score', 'Study_Hours_Per_Day', 'Sleep_Hours_Per_Day'],
        index=0
    )
    
    sorted_df = filtered_df.sort_values(by=sort_by, ascending=False).head(sample_size)
    
    # Create stacked bar chart for time allocation
    fig = go.Figure()
    
    for col in time_cols:
        fig.add_trace(go.Bar(
            y=sorted_df['Student_ID'],
            x=sorted_df[col],
            name=col.replace('_', ' ').replace('Per Day', ''),
            orientation='h'
        ))
    
    fig.update_layout(
        barmode='stack',
        title="Daily Time Distribution (Hours)",
        xaxis_title="Hours per Day",
        yaxis_title="Student ID",
        height=600,
        legend=dict(x=0.7, y=1.1, orientation='h')
    )
    
    # Add a line for 24 hours
    fig.add_shape(
        type="line",
        x0=24, y0=-0.5,
        x1=24, y1=len(sorted_df)-0.5,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=24, y=len(sorted_df)/2,
        text="24 Hours",
        showarrow=True,
        arrowhead=1,
        ax=30,
        ay=0
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart showing average time distribution
    avg_time = sorted_df[time_cols].mean().reset_index()
    avg_time.columns = ['Activity', 'Hours']
    
    # Calculate remaining hours
    total_hours = avg_time['Hours'].sum()
    if total_hours < 24:
        avg_time = pd.concat([avg_time, pd.DataFrame({'Activity': ['Other'], 'Hours': [24 - total_hours]})], ignore_index=True)
    
    avg_time['Activity'] = avg_time['Activity'].str.replace('_', ' ').str.replace('Per Day', '')
    
    fig = px.pie(avg_time, values='Hours', names='Activity', 
                title='Average Daily Time Distribution',
                hole=0.4)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Individual Student Profiles")
    
    # Student selector
    if len(filtered_df) > 0:
        selected_student = st.selectbox(
            "Select Student ID",
            options=sorted(filtered_df['Student_ID'].tolist())
        )
        
        student_data = filtered_df[filtered_df['Student_ID'] == selected_student].iloc[0]
        
        # Display student information
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.write("### Student Details")
            st.write(f"**Student ID:** {student_data['Student_ID']}")
            st.write(f"**GPA:** {student_data['GPA']}")
            st.write(f"**Stress Level:** {student_data['Stress_Level']}")
            st.write(f"**Work-Life Balance:** {student_data['Work_Life_Balance_Score']}")
        
        with col2:
            # Create a gauge for work-life balance
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = student_data['Work_Life_Balance_Score'],
                title = {'text': "Work-Life Balance"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.33], 'color': "lightcoral"},
                        {'range': [0.33, 0.66], 'color': "khaki"},
                        {'range': [0.66, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Create radar chart for student metrics
            radar_cols = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                         'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA', 'Work_Life_Balance_Score']
            
            # Normalize values for radar chart
            max_vals = {
                'Study_Hours_Per_Day': 12,
                'Sleep_Hours_Per_Day': 10,
                'Extracurricular_Hours_Per_Day': 6,
                'Social_Hours_Per_Day': 8,
                'Physical_Activity_Hours_Per_Day': 4,
                'GPA': 4,
                'Work_Life_Balance_Score': 1
            }
            
            radar_values = [student_data[col] / max_vals[col] for col in radar_cols]
            radar_labels = [col.replace('_', ' ').replace('Per Day', '') for col in radar_cols]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=radar_labels,
                fill='toself',
                name=student_data['Student_ID']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title="Student Profile (Normalized)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Time allocation for selected student
        st.subheader("Time Allocation")
        
        time_data = student_data[time_cols].reset_index()
        time_data.columns = ['Category', 'Hours']
        time_data['Category'] = time_data['Category'].apply(lambda x: x.replace('_', ' ').replace('Per Day', ''))
        
        # Calculate remaining hours
        total_hours = time_data['Hours'].sum()
        if total_hours < 24:
            time_data = pd.concat([time_data, pd.DataFrame({'Category': ['Other'], 'Hours': [24 - total_hours]})], ignore_index=True)
        
        fig = px.bar(time_data, x='Category', y='Hours', 
                    title="Daily Time Allocation",
                    color='Category',
                    text_auto=True)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display peer comparison
        st.subheader("Peer Comparison")
        
        # GPA comparison
        fig = go.Figure()
        
        # Add a histogram for all students
        fig.add_trace(go.Histogram(
            x=filtered_df['GPA'],
            name='All Students',
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        # Add vertical line for selected student
        fig.add_shape(
            type="line",
            x0=student_data['GPA'], y0=0,
            x1=student_data['GPA'], y1=30,
            line=dict(color="red", width=3)
        )
        
        fig.add_annotation(
            x=student_data['GPA'],
            y=25,
            text=f"{selected_student}: {student_data['GPA']}",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40
        )
        
        fig.update_layout(
            title="GPA Distribution with Student Comparison",
            xaxis_title="GPA",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Balance vs. GPA scatter plot with selected student highlighted
        fig = px.scatter(filtered_df, 
                        x='Work_Life_Balance_Score', 
                        y='GPA',
                        color='Stress_Level',
                        title="Work-Life Balance vs. GPA",
                        color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"})
        
        # Highlight selected student
        fig.add_trace(go.Scatter(
            x=[student_data['Work_Life_Balance_Score']],
            y=[student_data['GPA']],
            mode='markers',
            marker=dict(size=15, color='black', symbol='circle-open'),
            name=selected_student
        ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No students match the current filter criteria. Please adjust your filters.")

# Footer
st.markdown("---")
st.markdown("Student Life Balance Dashboard - Created with Streamlit")