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

# Load real data from CSV
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv('dataset.csv')
        
        # Add student ID for reference if not present
        if 'Student_ID' not in df.columns:
            df['Student_ID'] = [f'S{i+1:04d}' for i in range(len(df))]
        
        # Make sure stress level is properly categorized
        if 'Stress_Level' in df.columns:
            # Ensure values are properly formatted
            if df['Stress_Level'].dtype != 'object':
                # Map numerical values to string categories if needed
                df['Stress_Level'] = df['Stress_Level'].astype(str)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a small sample dataset as fallback
        sample_df = pd.DataFrame({
            'Study_Hours_Per_Day': [6.0, 7.5, 5.2, 8.1, 4.5],
            'Sleep_Hours_Per_Day': [7.0, 6.5, 8.0, 5.5, 7.5],
            'Extracurricular_Hours_Per_Day': [2.0, 1.5, 3.0, 1.0, 2.5],
            'Social_Hours_Per_Day': [3.0, 2.5, 4.0, 2.0, 3.5],
            'Physical_Activity_Hours_Per_Day': [1.5, 1.0, 2.0, 0.5, 1.0],
            'GPA': [3.5, 3.8, 3.2, 3.9, 3.0],
            'Stress_Level': ['Moderate', 'High', 'Low', 'High', 'Low'],
            'Work_Life_Balance_Score': [0.7, 0.6, 0.8, 0.5, 0.7],
            'Student_ID': ['S0001', 'S0002', 'S0003', 'S0004', 'S0005'],
        })
        return sample_df

# Load the data from CSV
df = load_data()

# Sidebar for filtering
st.sidebar.title("Student Life Balance Dashboard")

# Add number of students slider
num_students = st.sidebar.slider("Number of Students to Display", 
                              min_value=10, 
                              max_value=max(500, len(df)), 
                              value=min(50, len(df)),
                              step=10)

# Filtering options
st.sidebar.subheader("Filters")

# Stress level filter (multiselect)
stress_options = sorted(df['Stress_Level'].unique().tolist())
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

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Correlation Analysis", "Time Distribution", "Individual Students", "Data Explorer"])

with tab1:
    st.subheader("Student Life Balance Overview")
    
    # Stress Level Distribution
    stress_counts = filtered_df['Stress_Level'].value_counts().reset_index()
    stress_counts.columns = ['Stress_Level', 'Count']
    
    fig = px.pie(stress_counts, values='Count', names='Stress_Level', 
                title='Stress Level Distribution',
                color='Stress_Level',
                color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GPA Distribution by Stress Level
        fig = px.box(filtered_df, x="Stress_Level", y="GPA", 
                    color="Stress_Level", 
                    title="GPA Distribution by Stress Level",
                    color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"},
                    category_orders={"Stress_Level": ["Low", "Moderate", "High"]})
        fig.update_layout(height=400, boxmode='group', xaxis_title="Stress Level", yaxis_title="GPA")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Work-Life Balance by Stress Level
        fig = px.violin(filtered_df, x="Stress_Level", y="Work_Life_Balance_Score", 
                        color="Stress_Level", box=True,
                        title="Work-Life Balance Distribution by Stress Level",
                        color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"},
                        category_orders={"Stress_Level": ["Low", "Moderate", "High"]})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary of averages by stress level
    st.subheader("Averages by Stress Level")
    stress_summary = filtered_df.groupby('Stress_Level')[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
                                                       'GPA', 'Work_Life_Balance_Score']].mean().reset_index()
    
    fig = px.bar(stress_summary.melt(id_vars=['Stress_Level'], 
                                    value_vars=['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
                                                'GPA', 'Work_Life_Balance_Score']),
                x='Stress_Level', y='value', color='Stress_Level', facet_col='variable', 
                color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"},
                category_orders={"Stress_Level": ["Low", "Moderate", "High"]},
                title="Key Metrics by Stress Level")
    fig.update_layout(height=500)
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
    
    # Add annotation explaining bubble size
    fig.add_annotation(
        x=0.02,
        y=1.05,
        xref="paper",
        yref="paper",
        text="Bubble size represents Work-Life Balance Score",
        showarrow=False,
        font=dict(size=12),
        align="left",
        # bgcolor="rgba(255, 255, 255, 0.8)",
        # bordercolor="gray",
        # borderwidth=1,
        # borderpad=4
    )
    
    fig.update_layout(
        height=600,
        dragmode='zoom',
        legend=dict(
            title="Stress Level",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
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
    
    # Sort options
    sort_by = st.selectbox(
        "Sort Students By:",
        options=['Student_ID', 'GPA', 'Work_Life_Balance_Score', 'Study_Hours_Per_Day', 'Sleep_Hours_Per_Day'],
        index=0
    )
    
    # Handle case where filtered_df might be empty
    if len(filtered_df) > 0:
        sorted_df = filtered_df.sort_values(by=sort_by, ascending=False).head(num_students)
        
        # Create stacked bar chart for time allocation
        fig = go.Figure()
        
        # Define analogous color palette (blues to greens)
        analogous_colors = [
            '#1f77b4',  # Blue
            '#17becf',  # Light Blue
            '#2ca02c',  # Green
            '#bcbd22',  # Yellow-green
            '#ff7f0e'   # Orange
        ]
        
        for i, col in enumerate(time_cols):
            if col in sorted_df.columns:  # Make sure column exists
                fig.add_trace(go.Bar(
                    y=sorted_df['Student_ID'],
                    x=sorted_df[col],
                    name=col.replace('_', ' ').replace('Per Day', ''),
                    orientation='h',
                    marker_color=analogous_colors[i % len(analogous_colors)]
                ))
        
        fig.update_layout(
            barmode='stack',
            xaxis_title="Hours per Day",
            yaxis_title="Student ID",
            height=600,
            legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center'),
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 4, 8, 12, 16, 20, 24],
                range=[0, 24]
            ),
            showlegend=True
        )
        
        # Add centered title above the chart
        st.markdown("<h3 style='text-align: center;'>Daily Time Distribution (Hours)</h3>", unsafe_allow_html=True)
        
        # Center the chart using columns
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
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
                    hole=0.4,
                    color_discrete_sequence=['#1f77b4', '#17becf', '#2ca02c', '#bcbd22', '#ff7f0e', '#d62728'])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available with current filter settings. Please adjust your filters.")

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
            
            # Filter for columns that exist in the dataset
            radar_cols = [col for col in radar_cols if col in student_data.index]
            
            if radar_cols:
                # Find the maximum hours across all hour columns in the entire dataset
                hour_columns = [col for col in filtered_df.columns if 'Hours_Per_Day' in col]
                if hour_columns:
                    # Get the global maximum hours across all types of activities
                    max_hours = filtered_df[hour_columns].values.max()
                else:
                    max_hours = 12.0  # Fallback if no hour columns found
                
                # Apply consistent normalization rules for each type of data
                radar_values = []
                radar_text = []
                
                for col in radar_cols:
                    value = student_data[col]
                    
                    # Apply different normalization based on data type
                    if 'Hours_Per_Day' in col:
                        # Normalize all hour columns by the same global maximum
                        normalized_value = value / max_hours
                        radar_text.append(f"{value:.1f}h<br>({normalized_value:.0%})")
                    elif col == 'GPA':
                        # For GPA, normalize by dividing by 4
                        normalized_value = value / 4.0
                        radar_text.append(f"{value:.2f}<br>({normalized_value:.0%})")
                    elif col == 'Work_Life_Balance_Score':
                        # For Work-life balance, normalize by dividing by 1
                        normalized_value = value / 1.0
                        radar_text.append(f"{value:.2f}<br>({normalized_value:.0%})")
                    else:
                        # For any other metric, use a safe normalization
                        max_val = filtered_df[col].max()
                        normalized_value = value / max_val
                        radar_text.append(f"{value:.2f}<br>({normalized_value:.0%})")
                    
                    radar_values.append(normalized_value)
                        
                radar_labels = [col.replace('_', ' ').replace('Per Day', '') for col in radar_cols]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=radar_labels,
                    fill='toself',
                    name=student_data['Student_ID'],
                    text=radar_text,
                    hoverinfo="text+name"
                ))
                
                # Set radial axis to show percentages
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickvals=[0, 0.25, 0.5, 0.75, 1],
                            ticktext=["0%", "25%", "50%", "75%", "100%"]
                        )
                    ),
                    showlegend=False,
                    title="Student Profile (Normalized)"
                )
                
                # Add annotation explaining normalization
                normalization_text = f"Hours normalized to {max_hours:.1f}hr max | GPA to 4.0 max | Work-Life Balance to 1.0 max"
                fig.add_annotation(
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    text=normalization_text,
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Time allocation for selected student
        st.subheader("Time Allocation")
        
        time_columns = [col for col in time_cols if col in student_data.index]
        if time_columns:
            time_data = pd.DataFrame({
                'Category': [col.replace('_', ' ').replace('Per Day', '') for col in time_columns],
                'Hours': [student_data[col] for col in time_columns]
            })
            
            # Calculate remaining hours
            total_hours = time_data['Hours'].sum()
            if total_hours < 24:
                time_data = pd.concat([time_data, pd.DataFrame({'Category': ['Other'], 'Hours': [24 - total_hours]})], ignore_index=True)
            
            fig = px.bar(time_data, x='Category', y='Hours', 
                        title="Daily Time Allocation",
                        text_auto=True,
                        color='Category',
                        color_discrete_sequence=['#1f77b4', '#17becf', '#2ca02c', '#bcbd22', '#ff7f0e', '#d62728'])
            
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
        
        # Highlight selected student with a high contrast marker
        fig.add_trace(go.Scatter(
            x=[student_data['Work_Life_Balance_Score']],
            y=[student_data['GPA']],
            mode='markers',
            marker=dict(
                size=18, 
                color='white',  # White border for contrast
                line=dict(
                    color='lime',  # Bright color for visibility against dark background
                    width=3
                ),
                symbol='circle-open'
            ),
            name=selected_student
        ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No students match the current filter criteria. Please adjust your filters.")

with tab5:
    st.subheader("Data Explorer")
    
    # Show a sample of the data
    st.write("### Data Sample")
    
    # Column selector for the data viewer
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to view",
        options=all_columns,
        default=all_columns[:6]  # Show first 6 columns by default
    )
    
    # Filter data based on selected columns
    if selected_columns:
        st.dataframe(filtered_df[selected_columns].head(num_students))
    else:
        st.dataframe(filtered_df.head(num_students))
    
    # Data summary
    st.write("### Data Summary")
    
    # Summary statistics
    st.write("#### Summary Statistics")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_columns:
        summary_stats = filtered_df[numeric_columns].describe()
        st.dataframe(summary_stats)
    
    # Histogram for selected feature
    st.write("#### Distribution of Selected Feature")
    hist_column = st.selectbox(
        "Select column for histogram",
        options=numeric_columns,
        index=0
    )
    
    fig = px.histogram(
        filtered_df, 
        x=hist_column,
        color="Stress_Level",
        marginal="box",
        title=f"Distribution of {hist_column}",
        color_discrete_map={"Low": "green", "Moderate": "gold", "High": "red"}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download data button
    st.write("### Download Filtered Data")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        "Download CSV",
        csv,
        "filtered_student_data.csv",
        "text/csv",
        key='download-csv'
    )

# Footer
st.markdown("---")
st.markdown("Student Life Balance Dashboard - Created with Streamlit")