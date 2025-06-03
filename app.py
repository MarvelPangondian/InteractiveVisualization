import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

activity_colors = {
    "Study_Hours_Per_Day": "#4e79a7",
    "Sleep_Hours_Per_Day": "#a0cbe8",
    "Physical_Activity_Hours_Per_Day": "#f28e2b",
    "Social_Hours_Per_Day": "#e15759",
    "Extracurricular_Hours_Per_Day": "#76b7b2",
}
category_orders_stress = {"Stress_Level": ["Low", "Moderate", "High"]}
color_discrete_map_stress = {
    "Low": "#2E8B57",
    "Moderate": "#FFD700",
    "High": "#DC143C",
}

st.set_page_config(
    page_title="Student Life Balance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    @media (max-width: 768px) {
        .metric-container {
            flex-direction: column;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load real data from CSV
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv("dataset.csv")

        # Add student ID for reference if not present
        if "Student_ID" not in df.columns:
            df["Student_ID"] = [f"S{i+1:04d}" for i in range(len(df))]

        # Ensure proper data types
        if "Stress_Level" in df.columns:
            if df["Stress_Level"].dtype != "object":
                df["Stress_Level"] = df["Stress_Level"].astype(str)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data as fallback
        np.random.seed(42)
        n_students = 100

        sample_df = pd.DataFrame(
            {
                "Study_Hours_Per_Day": np.random.normal(7, 2, n_students).clip(2, 12),
                "Sleep_Hours_Per_Day": np.random.normal(7.5, 1.5, n_students).clip(
                    4, 10
                ),
                "Extracurricular_Hours_Per_Day": np.random.normal(
                    2, 1, n_students
                ).clip(0, 5),
                "Social_Hours_Per_Day": np.random.normal(3, 1.5, n_students).clip(0, 6),
                "Physical_Activity_Hours_Per_Day": np.random.normal(
                    1.5, 0.8, n_students
                ).clip(0, 4),
                "GPA": np.random.normal(3.2, 0.5, n_students).clip(2.0, 4.0),
                "Stress_Level": np.random.choice(
                    ["Low", "Moderate", "High"], n_students, p=[0.3, 0.4, 0.3]
                ),
                "Work_Life_Balance_Score": np.random.beta(2, 2, n_students),
                "Student_ID": [f"S{i+1:04d}" for i in range(n_students)],
            }
        )

        # Add numeric stress level
        stress_map = {"Low": 1, "Moderate": 2, "High": 3}
        sample_df["Stress_Level_Num"] = sample_df["Stress_Level"].map(stress_map)

        return sample_df


# Advanced analytics functions
@st.cache_data
def perform_clustering(df, n_clusters=3):
    """Perform K-means clustering on student data"""
    features = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "GPA",
        "Work_Life_Balance_Score",
    ]
    X = df[features].fillna(df[features].mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    return clusters, kmeans, scaler


@st.cache_data
def calculate_correlations(df):
    """Calculate correlation matrix and significance"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Calculate p-values
    n = len(df)
    p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                r = corr_matrix.iloc[i, j]
                t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                p_values.iloc[i, j] = p_val
            else:
                p_values.iloc[i, j] = 0

    return corr_matrix, p_values


def add_regression_line(fig, x_data, y_data, name="Trend"):
    """Add regression line to scatter plot"""
    x_clean = pd.Series(x_data).dropna()
    y_clean = pd.Series(y_data).dropna()

    if len(x_clean) > 1 and len(y_clean) > 1:
        # Align the data
        aligned_data = pd.DataFrame({"x": x_data, "y": y_data}).dropna()
        if len(aligned_data) > 1:
            X = aligned_data["x"].values.reshape(-1, 1)
            y = aligned_data["y"].values

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)

            # Sort for line plotting
            sort_idx = np.argsort(aligned_data["x"])
            x_sorted = aligned_data["x"].iloc[sort_idx]
            y_pred_sorted = y_pred[sort_idx]

            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_pred_sorted,
                    mode="lines",
                    name=f"{name} (R² = {r2:.3f})",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

    return fig


# Load the data
df = load_data()

# Sidebar for filtering
st.sidebar.title("🎛️ Dashboard Controls")

# Filter presets
st.sidebar.subheader("📋 Quick Filters")
filter_preset = st.sidebar.selectbox(
    "Choose a preset filter:",
    ["Custom", "High Achievers", "Well Balanced", "High Stress"],
)

# Apply preset filters
if filter_preset == "High Achievers":
    gpa_min, gpa_max = 3.5, df["GPA"].max()
    stress_default = ["Low", "Moderate"]
    balance_min, balance_max = 0.7, df["Work_Life_Balance_Score"].max()
elif filter_preset == "Well Balanced":
    gpa_min, gpa_max = df["GPA"].min(), df["GPA"].max()
    stress_default = ["Low", "Moderate", "High"]
    balance_min, balance_max = 0.6, df["Work_Life_Balance_Score"].max()
elif filter_preset == "High Stress":
    gpa_min, gpa_max = df["GPA"].min(), df["GPA"].max()
    stress_default = ["High"]
    balance_min, balance_max = (
        df["Work_Life_Balance_Score"].min(),
        df["Work_Life_Balance_Score"].max(),
    )
else:  # Custom
    gpa_min, gpa_max = df["GPA"].min(), df["GPA"].max()
    stress_default = df["Stress_Level"].unique().tolist()
    balance_min, balance_max = (
        df["Work_Life_Balance_Score"].min(),
        df["Work_Life_Balance_Score"].max(),
    )

# Individual filter controls
st.sidebar.subheader("🔧 Custom Filters")

# Number of students slider
num_students = st.sidebar.slider(
    "Number of Students to Display",
    min_value=10,
    max_value=max(500, len(df)),
    value=min(100, len(df)),
    step=10,
)

# Stress level filter
stress_options = sorted(df["Stress_Level"].unique().tolist())
selected_stress = st.sidebar.multiselect(
    "Stress Level", options=stress_options, default=stress_default
)
if not selected_stress:
    selected_stress = stress_options

# GPA range filter
gpa_range = st.sidebar.slider(
    "GPA Range",
    min_value=float(df["GPA"].min()),
    max_value=float(df["GPA"].max()),
    value=(float(gpa_min), float(gpa_max)),
    step=0.1,
)

# Balance score filter
balance_range = st.sidebar.slider(
    "Work-Life Balance Score",
    min_value=float(df["Work_Life_Balance_Score"].min()),
    max_value=float(df["Work_Life_Balance_Score"].max()),
    value=(float(balance_min), float(balance_max)),
    step=0.05,
)

# Advanced options
st.sidebar.subheader("⚙️ Advanced Options")
show_regression = st.sidebar.checkbox("Show Regression Lines", value=True)
show_clusters = st.sidebar.checkbox("Show Student Clusters", value=False)
show_statistical_info = st.sidebar.checkbox(
    "Show Statistical Significance", value=False
)

# Apply filters
filtered_df = df[
    (df["Stress_Level"].isin(selected_stress))
    & (df["GPA"] >= gpa_range[0])
    & (df["GPA"] <= gpa_range[1])
    & (df["Work_Life_Balance_Score"] >= balance_range[0])
    & (df["Work_Life_Balance_Score"] <= balance_range[1])
].head(num_students)

# Perform clustering if enabled
if show_clusters and len(filtered_df) > 3:
    clusters, kmeans_model, scaler = perform_clustering(filtered_df)
    filtered_df = filtered_df.copy()
    filtered_df["Cluster"] = [f"Group {i+1}" for i in clusters]

# Main dashboard
st.title("🎓 Student Life Balance Analytics Dashboard - V2")
st.markdown(
    f"**Analyzing data for {len(filtered_df)} students** | *Dataset: {len(df)} total students*"
)

# Key insights box
if len(filtered_df) > 0:
    avg_gpa = filtered_df["GPA"].mean()
    high_stress_pct = (filtered_df["Stress_Level"] == "High").mean() * 100
    avg_balance = filtered_df["Work_Life_Balance_Score"].mean()

    insights = []
    if avg_gpa >= 3.5:
        insights.append("🎯 **High Performance Group**: Above average GPA detected.")
    if high_stress_pct > 60:
        insights.append(
            "⚠️ **High Stress Alert**: Over 60% of students showing high stress levels."
        )
    if avg_balance >= 0.7:
        insights.append(
            "✅ **Good Balance**: Students showing healthy work-life balance."
        )

    if insights:
        for insight in insights:
            st.info(insight)

col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_gpa = filtered_df["GPA"].mean()
    gpa_change = avg_gpa - df["GPA"].mean()
    st.metric("Average GPA", f"{avg_gpa:.2f}")

with col2:
    avg_study = filtered_df["Study_Hours_Per_Day"].mean()
    study_change = avg_study - df["Study_Hours_Per_Day"].mean()
    st.metric("Average Study Hours", f"{avg_study:.1f}")

with col3:
    avg_sleep = filtered_df["Sleep_Hours_Per_Day"].mean()
    sleep_change = avg_sleep - df["Sleep_Hours_Per_Day"].mean()
    st.metric("Average Sleep Hours", f"{avg_sleep:.1f}")

with col4:
    avg_balance = filtered_df["Work_Life_Balance_Score"].mean()
    balance_change = avg_balance - df["Work_Life_Balance_Score"].mean()
    st.metric("Average Balance Score", f"{avg_balance:.2f}")

# tabs with better organization
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Overview",
        "🔗 Correlation Analysis",
        "⏰ Time Distribution",
        "👤 Individual Students",
        "🧩 Advanced Analytics",
        "📋 Data Explorer",
    ]
)

with tab1:
    st.subheader("Student Life Balance Overview")

    col1, col2 = st.columns(2)

    with col1:
        # stress level distribution
        stress_counts = filtered_df["Stress_Level"].value_counts().reset_index()
        stress_counts.columns = ["Stress_Level", "Count"]

        fig = px.pie(
            stress_counts,
            values="Count",
            names="Stress_Level",
            title="Stress Level Distribution",
            color="Stress_Level",
            color_discrete_map=color_discrete_map_stress,
            category_orders=category_orders_stress,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label+value")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # GPA vs Stress with styling
        fig = px.box(
            filtered_df,
            x="Stress_Level",
            y="GPA",
            color="Stress_Level",
            title="GPA Distribution by Stress Level",
            color_discrete_map=color_discrete_map_stress,
            category_orders=category_orders_stress,
        )

        # Add statistical annotations
        if show_statistical_info:
            for stress in ["Low", "Moderate", "High"]:
                subset = filtered_df[filtered_df["Stress_Level"] == stress]["GPA"]
                if len(subset) > 0:
                    fig.add_annotation(
                        x=stress,
                        y=subset.max() + 0.1,
                        text=f"μ={subset.mean():.2f}<br>σ={subset.std():.2f}",
                        showarrow=False,
                        font=dict(size=10),
                    )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Averages by Stress Level")

    # Define the metrics to compare
    metrics = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "GPA",
        "Work_Life_Balance_Score",
    ]

    # Create a tidy DataFrame for Plotly
    avg_by_stress = filtered_df.groupby("Stress_Level")[metrics].mean().reset_index()
    melted_df = avg_by_stress.melt(
        id_vars="Stress_Level", var_name="Metric", value_name="Value"
    )

    # Plot with Plotly Express
    fig = px.bar(
        melted_df,
        x="Stress_Level",
        y="Value",
        color="Stress_Level",
        facet_col="Metric",
        category_orders=category_orders_stress,
        color_discrete_map=color_discrete_map_stress,
        title="Average Metrics by Stress Level",
    )

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🔗 Correlation Analysis")

    # Variable selection with better defaults
    col1, col2 = st.columns(2)

    numeric_columns = [
        col for col in filtered_df.columns if col not in ["Student_ID", "Stress_Level"]
    ]

    with col1:
        x_var = st.selectbox(
            "Select X-axis Variable",
            options=numeric_columns,
            index=0 if "Study_Hours_Per_Day" in numeric_columns else 0,
        )

    with col2:
        y_var = st.selectbox(
            "Select Y-axis Variable",
            options=numeric_columns,
            index=numeric_columns.index("GPA") if "GPA" in numeric_columns else 1,
        )

    # scatter plot with multiple options
    if show_clusters and "Cluster" in filtered_df.columns:
        color_var = "Cluster"
        color_map = None
    else:
        color_var = "Stress_Level"
        color_map = color_discrete_map_stress

    fig = px.scatter(
        filtered_df,
        x=x_var,
        y=y_var,
        color=color_var,
        size="Work_Life_Balance_Score",
        hover_name="Student_ID",
        title=f"Relationship: {x_var.replace('_', ' ')} vs {y_var.replace('_', ' ')}",
        color_discrete_map=color_map,
        category_orders=category_orders_stress,
        size_max=15,
        opacity=0.7,
    )

    # Add regression line if enabled
    if show_regression:
        fig = add_regression_line(fig, filtered_df[x_var], filtered_df[y_var])

    # layout
    fig.update_layout(
        height=600,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add correlation coefficient
    corr_coef = filtered_df[x_var].corr(filtered_df[y_var])
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Correlation: r = {corr_coef:.3f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    st.plotly_chart(fig, use_container_width=True)

    # correlation matrix
    st.subheader("📊 Correlation Matrix")

    # Calculate correlations with significance
    filtered_second = filtered_df.drop(
        columns=[
            col
            for col in filtered_df.columns
            if filtered_df[col].nunique(dropna=True) <= 1
        ]
    )
    corr_matrix, p_values = calculate_correlations(filtered_second)

    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    if show_statistical_info:
        st.caption("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")

with tab3:
    st.subheader("⏰ Daily Time Distribution")

    # Time allocation analysis
    time_cols = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
    ]

    # Filter for existing columns
    available_time_cols = [col for col in time_cols if col in filtered_df.columns]

    if available_time_cols:
        # Sort options
        sort_options = [
            "Student_ID",
            "GPA",
            "Work_Life_Balance_Score",
        ] + available_time_cols
        sort_by = st.selectbox(
            "Sort Students By:", options=sort_options, index=1  # Default to GPA
        )

        # Display options
        chart_height = st.slider("Chart height:", 400, 800, 600)

        if len(filtered_df) > 0:
            sorted_df = filtered_df.sort_values(by=sort_by, ascending=False).head(
                num_students
            )

            # Create enhanced stacked bar chart
            fig = go.Figure()

            colors = ["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e"]

            for col in available_time_cols:
                fig.add_trace(
                    go.Bar(
                        y=sorted_df["Student_ID"],
                        x=sorted_df[col],
                        name=col.replace("_", " ").replace("Per Day", "").title(),
                        orientation="h",
                        marker_color=activity_colors.get(col, "#888"),  # fallback: gray
                        hovertemplate=f'<b>%{{y}}</b><br>{col.replace("_", " ")}: %{{x:.1f}} hours<extra></extra>',
                    )
                )

            fig.update_layout(
                barmode="stack",
                title="Daily Time Distribution by Student",
                xaxis_title="Hours per Day",
                yaxis_title="Student ID",
                height=chart_height,
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(range=[0, 24]),
            )

            # Add 24-hour reference line
            fig.add_vline(
                x=24,
                line_dash="dash",
                line_color="red",
                annotation_text="24 hours",
                annotation_position="top",
            )

            st.plotly_chart(fig, use_container_width=True)

            # time distribution pie chart
            avg_time = sorted_df[available_time_cols].mean()
            total_tracked = avg_time.sum()

            if total_tracked < 24:
                avg_time["Other/Free Time"] = 24 - total_tracked

            # Create donut chart
            pie_labels = [
                name.replace("_", " ").replace("Per Day", "").title()
                for name in avg_time.index
            ]
            pie_colors = [activity_colors.get(col, "#888") for col in avg_time.index]

            fig = px.pie(
                values=avg_time.values,
                names=pie_labels,
                title="Average Daily Time Distribution",
                hole=0.4,
                color_discrete_sequence=pie_colors,
            )

            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("👤 Individual Student Profiles")

    if len(filtered_df) > 0:
        # student selector
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_student = st.selectbox(
                "Select Student ID",
                options=sorted(df["Student_ID"].tolist()),
                help="Choose a student to view detailed profile",
            )

        with col2:
            show_peer_percentile = st.checkbox("Show Peer Percentiles", value=True)

        student_data = df[df["Student_ID"] == selected_student].iloc[0]

        # Student details layout
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

        with col1:
            st.write("### 📋 Student Details")
            st.write(f"**Student ID:** {student_data['Student_ID']}")
            st.write(f"**GPA:** {student_data['GPA']:.2f}")
            st.write(f"**Stress Level:** {student_data['Stress_Level']}")
            st.write(
                f"**Work-Life Balance:** {student_data['Work_Life_Balance_Score']:.3f}"
            )

            if show_clusters and "Cluster" in student_data:
                st.write(f"**Student Group:** {student_data['Cluster']}")

            if show_peer_percentile:
                st.write("### 📊 Peer Rankings")
                gpa_percentile = (
                    student_data["GPA"] >= filtered_df["GPA"]
                ).mean() * 100
                balance_percentile = (
                    student_data["Work_Life_Balance_Score"]
                    >= filtered_df["Work_Life_Balance_Score"]
                ).mean() * 100
                st.write(f"**GPA Percentile:** {gpa_percentile:.0f}%")
                st.write(f"**Balance Percentile:** {balance_percentile:.0f}%")

        with col2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=student_data["Work_Life_Balance_Score"],
                    title={
                        "text": "<b>Work-Life Balance Score</b>",
                        "font": {"size": 16},
                    },
                    delta={
                        "reference": filtered_df["Work_Life_Balance_Score"].mean(),
                        "increasing": {"color": "green"},
                        "decreasing": {"color": "red"},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 1],
                            "tickwidth": 1,
                            "tickcolor": "darkgray",
                        },
                        "bar": {"color": "navy"},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "gray",
                        "steps": [
                            {"range": [0, 0.33], "color": "lightcoral"},
                            {"range": [0.33, 0.66], "color": "khaki"},
                            {"range": [0.66, 1], "color": "lightgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": filtered_df["Work_Life_Balance_Score"].mean(),
                        },
                    },
                    number={
                        "font": {"size": 28},  # Control number font size
                        "suffix": "",  # No suffix like % unless needed
                    },
                    domain={"x": [0, 1], "y": [0, 1]},
                )
            )
            fig.update_layout(height=300, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            radar_cols = [
                "Study_Hours_Per_Day",
                "Sleep_Hours_Per_Day",
                "Extracurricular_Hours_Per_Day",
                "Social_Hours_Per_Day",
                "Physical_Activity_Hours_Per_Day",
                "GPA",
                "Work_Life_Balance_Score",
            ]
            available_radar_cols = [
                col for col in radar_cols if col in student_data.index
            ]

            if available_radar_cols:
                radar_values, radar_labels, radar_text = [], [], []

                for col in available_radar_cols:
                    value = student_data[col]
                    if "Hours_Per_Day" in col:
                        max_val = 12.0
                        normalized_value = min(value / max_val, 1.0)
                        radar_text.append(f"{value:.1f}h")
                    elif col == "GPA":
                        normalized_value = value / 4.0
                        radar_text.append(f"{value:.2f}")
                    elif col == "Work_Life_Balance_Score":
                        normalized_value = value
                        radar_text.append(f"{value:.2f}")
                    else:
                        max_val = filtered_df[col].max()
                        normalized_value = value / max_val if max_val > 0 else 0
                        radar_text.append(f"{value:.2f}")

                    radar_values.append(normalized_value)
                    radar_labels.append(
                        col.replace("_", " ").replace("Per Day", "").title()
                    )

                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=radar_values,
                        theta=radar_labels,
                        fill="toself",
                        name=student_data["Student_ID"],
                        line_color="blue",
                        fillcolor="rgba(0, 100, 255, 0.3)",
                        hovertemplate="%{theta}: %{text}<extra></extra>",
                        text=radar_text,
                    )
                )

                avg_values = []
                for col in available_radar_cols:
                    avg_val = filtered_df.head(num_students)[col].mean()
                    if "Hours_Per_Day" in col:
                        normalized_avg = min(avg_val / 12.0, 1.0)
                    elif col == "GPA":
                        normalized_avg = avg_val / 4.0
                    elif col == "Work_Life_Balance_Score":
                        normalized_avg = avg_val
                    else:
                        max_val = filtered_df[col].max()
                        normalized_avg = avg_val / max_val if max_val > 0 else 0
                    avg_values.append(normalized_avg)

                fig.add_trace(
                    go.Scatterpolar(
                        r=avg_values,
                        theta=radar_labels,
                        fill="toself",
                        name="Display Group Average",
                        line_color="red",
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        opacity=0.6,
                    )
                )

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickvals=[0, 0.25, 0.5, 0.75, 1],
                            ticktext=["0%", "25%", "50%", "75%", "100%"],
                        )
                    ),
                    showlegend=True,
                    title="Student Profile Comparison",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("⏰ Individual Time Allocation")
        time_columns = [col for col in available_time_cols if col in student_data.index]
        if time_columns:
            col1, col2 = st.columns(2)
            with col1:
                time_data = pd.DataFrame(
                    {
                        "Category": [
                            col.replace("_", " ").replace("Per Day", "").title()
                            for col in time_columns
                        ],
                        "Hours": [student_data[col] for col in time_columns],
                    }
                )
                fig = px.bar(
                    time_data,
                    x="Category",
                    y="Hours",
                    title="Daily Time Allocation",
                    text="Hours",
                    color="Hours",
                    color_continuous_scale="viridis",
                )
                fig.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("### 💡 Personalized Recommendations")
                recommendations = []
                if student_data["Study_Hours_Per_Day"] > 9:
                    recommendations.append(
                        "⚠️ Consider reducing study hours to prevent burnout"
                    )
                elif student_data["Study_Hours_Per_Day"] < 4:
                    recommendations.append(
                        "📚 Consider increasing study time for better performance"
                    )
                if student_data["Sleep_Hours_Per_Day"] < 6:
                    recommendations.append(
                        "😴 Prioritize getting more sleep (7-9 hours recommended)"
                    )
                elif student_data["Sleep_Hours_Per_Day"] > 9:
                    recommendations.append("⏰ Consider optimizing sleep schedule")
                if student_data["Work_Life_Balance_Score"] < 0.5:
                    recommendations.append("⚖️ Focus on improving work-life balance")
                if student_data["Stress_Level"] == "High":
                    recommendations.append(
                        "🧘 High stress detected - consider stress management techniques"
                    )
                if (
                    "Physical_Activity_Hours_Per_Day" in student_data
                    and student_data["Physical_Activity_Hours_Per_Day"] < 1
                ):
                    recommendations.append(
                        "🏃 Increase physical activity for better well-being"
                    )
                if not recommendations:
                    recommendations.append("✅ Great balance! Keep up the good work!")
                for rec in recommendations:
                    st.write(f"• {rec}")

        # ✅ Add GPA Distribution Chart
        st.subheader("📊 Peer GPA Comparison")
        fig = go.Figure()
        hist_values, _ = np.histogram(filtered_df["GPA"], bins=10)
        max_count = max(hist_values)

        fig.add_trace(
            go.Histogram(
                x=filtered_df["GPA"],
                name="All Students",
                opacity=0.75,
                marker_color="lightblue",
            )
        )

        fig.add_shape(
            type="line",
            x0=student_data["GPA"],
            y0=0,
            x1=student_data["GPA"],
            y1=max_count + 2,
            line=dict(color="red", width=3),
        )

        fig.add_annotation(
            x=student_data["GPA"],
            y=max_count + 1,
            text=f"{selected_student}: {student_data['GPA']:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40,
        )

        fig.update_layout(
            title="GPA Distribution with Student Highlight",
            xaxis_title="GPA",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add Balance vs GPA with Highlight
        st.subheader("⚖️ Work-Life Balance vs GPA")
        fig = px.scatter(
            filtered_df,
            x="Work_Life_Balance_Score",
            y="GPA",
            color="Stress_Level",
            title="Work-Life Balance vs GPA",
            color_discrete_map=color_discrete_map_stress,
        )

        fig.add_trace(
            go.Scatter(
                x=[student_data["Work_Life_Balance_Score"]],
                y=[student_data["GPA"]],
                mode="markers",
                marker=dict(
                    size=16,
                    color="white",
                    line=dict(color="lime", width=3),
                    symbol="circle-open",
                ),
                name=selected_student,
            )
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🧩 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🎯 Student Clustering Analysis")

        if show_clusters and len(filtered_df) > 3:
            # Display cluster information
            cluster_summary = (
                filtered_df.groupby("Cluster")
                .agg(
                    {
                        "GPA": "mean",
                        "Study_Hours_Per_Day": "mean",
                        "Work_Life_Balance_Score": "mean",
                        "Stress_Level": lambda x: (
                            x.mode().iloc[0] if len(x.mode()) > 0 else "Mixed"
                        ),
                    }
                )
                .round(2)
            )

            st.write("**Cluster Characteristics:**")
            st.dataframe(cluster_summary)

            # Cluster visualization
            features_for_viz = ["Study_Hours_Per_Day", "GPA"]
            if all(col in filtered_df.columns for col in features_for_viz):
                fig = px.scatter(
                    filtered_df,
                    x=features_for_viz[0],
                    y=features_for_viz[1],
                    color="Cluster",
                    title="Student Clusters",
                    size="Work_Life_Balance_Score",
                    hover_data=["Student_ID", "Stress_Level"],
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Enable 'Show Student Clusters' in the sidebar to see clustering analysis"
            )

    with col2:
        st.write("### 📊 Statistical Analysis")

        # ANOVA test for stress levels and GPA
        if len(filtered_df["Stress_Level"].unique()) > 1:
            try:
                stress_groups = [
                    group["GPA"].values
                    for name, group in filtered_df.groupby("Stress_Level")
                ]
                if all(len(group) > 0 for group in stress_groups):
                    f_stat, p_value = stats.f_oneway(*stress_groups)

                    st.write("**ANOVA Test (Stress vs GPA):**")
                    st.write(f"• F-statistic: {f_stat:.3f}")
                    st.write(f"• p-value: {p_value:.3f}")

                    if p_value < 0.05:
                        st.write("• ✅ Significant difference between stress groups")
                    else:
                        st.write("• ❌ No significant difference between stress groups")
                else:
                    st.write("Insufficient data for ANOVA test")
            except Exception as e:
                st.write("Unable to perform ANOVA test")

        # Correlation significance tests
        st.write("**Key Correlations:**")
        key_pairs = [
            ("Study_Hours_Per_Day", "GPA"),
            ("Sleep_Hours_Per_Day", "GPA"),
            ("Work_Life_Balance_Score", "GPA"),
        ]

        for var1, var2 in key_pairs:
            if var1 in filtered_df.columns and var2 in filtered_df.columns:
                try:
                    corr_coef, p_val = stats.pearsonr(
                        filtered_df[var1], filtered_df[var2]
                    )
                    significance = (
                        "***"
                        if p_val < 0.001
                        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    )
                    st.write(
                        f"• {var1.replace('_', ' ')} ↔ {var2}: r={corr_coef:.3f}{significance}"
                    )
                except:
                    st.write(
                        f"• {var1.replace('_', ' ')} ↔ {var2}: Unable to calculate"
                    )

    # Parallel coordinates plot
    st.subheader("🌐 Parallel Coordinates Analysis")

    parallel_cols = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "GPA",
        "Work_Life_Balance_Score",
    ]
    available_parallel_cols = [
        col for col in parallel_cols if col in filtered_df.columns
    ]

    if len(available_parallel_cols) >= 3:
        try:
            # Sample data if too many points
            plot_df = filtered_df.sample(n=min(200, len(filtered_df)), random_state=42)
            plot_df_temp = plot_df.copy()

            stress_map = {"Low": 1, "Moderate": 2, "High": 3}
            plot_df_temp["Stress_Numeric"] = plot_df_temp["Stress_Level"].map(
                stress_map
            )
            custom_colorscale = [
                [0.0, "#2E8B57"],  # Low
                [0.5, "#FFD700"],  # Moderate
                [1.0, "#DC143C"],  # High
            ]

            fig = px.parallel_coordinates(
                plot_df_temp,
                dimensions=available_parallel_cols,
                color="Stress_Numeric",
                color_continuous_scale=custom_colorscale,
                title="Multi-dimensional Student Profile Analysis",
                range_color=[1, 3],  # Ensure the full 1–3 range is covered
            )
            fig.update_layout(
                height=500,
                margin=dict(l=100, r=40, t=150, b=40),  # increased left margin
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Each line represents a student. Stress level is encoded by color: Green=Low, Yellow=Moderate, Red=High."
            )
        except Exception as e:
            st.error("Unable to create parallel coordinates plot")

    # Predictive insights
    st.subheader("🔮 Predictive Insights")

    if len(filtered_df) > 10:
        # Simple linear regression for GPA prediction
        features = [
            "Study_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Work_Life_Balance_Score",
        ]
        available_features = [f for f in features if f in filtered_df.columns]

        if len(available_features) >= 2 and "GPA" in filtered_df.columns:
            try:
                X = filtered_df[available_features].fillna(
                    filtered_df[available_features].mean()
                )
                y = filtered_df["GPA"]

                reg_model = LinearRegression().fit(X, y)
                r2 = reg_model.score(X, y)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**GPA Prediction Model:**")
                    st.write(f"• R² Score: {r2:.3f}")
                    st.write("• Feature Importance:")
                    for feature, coef in zip(available_features, reg_model.coef_):
                        st.write(f"  - {feature.replace('_', ' ')}: {coef:.3f}")

                with col2:
                    st.write("**What-if Analysis:**")
                    st.write("Adjust values to see predicted GPA:")

                    # Simple what-if calculator
                    input_values = []

                    if "Study_Hours_Per_Day" in available_features:
                        study_input = st.slider(
                            "Study Hours:", 2.0, 12.0, 7.0, 0.5, key="study_pred"
                        )
                        input_values.append(study_input)

                    if "Sleep_Hours_Per_Day" in available_features:
                        sleep_input = st.slider(
                            "Sleep Hours:", 4.0, 10.0, 7.5, 0.5, key="sleep_pred"
                        )
                        input_values.append(sleep_input)

                    if "Work_Life_Balance_Score" in available_features:
                        balance_input = st.slider(
                            "Balance Score:", 0.0, 1.0, 0.7, 0.1, key="balance_pred"
                        )
                        input_values.append(balance_input)

                    # Predict GPA
                    if len(input_values) == len(available_features):
                        predicted_gpa = reg_model.predict([input_values])[0]
                        st.write(f"**Predicted GPA: {predicted_gpa:.2f}**")

                        # Show confidence
                        if predicted_gpa >= 3.5:
                            st.success("Excellent predicted performance!")
                        elif predicted_gpa >= 3.0:
                            st.info("Good predicted performance")
                        else:
                            st.warning("Consider adjusting study habits")
            except Exception as e:
                st.error("Unable to create prediction model")
        else:
            st.info("Insufficient features for prediction model")
    else:
        st.info("Need more data points for predictive analysis")

with tab6:
    st.subheader("📋 Data Explorer")

    # data sample viewer
    st.write("### 🔍 Data Sample")

    col1, col2 = st.columns(2)

    with col1:
        # Column selector
        all_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to view",
            options=all_columns,
            default=all_columns[:6] if len(all_columns) >= 6 else all_columns,
        )

    with col2:
        # Display options
        show_raw_data = st.checkbox("Show raw data", value=True)
        show_summary_stats = st.checkbox("Show summary statistics", value=True)

    if selected_columns and show_raw_data:
        # data display with search
        search_term = st.text_input(
            "🔍 Search in data (Student ID, Stress Level, etc.):"
        )

        display_df = filtered_df[selected_columns].copy()

        if search_term:
            # Simple search across string columns
            mask = pd.Series([False] * len(display_df))
            for col in display_df.columns:
                if display_df[col].dtype == "object":
                    mask |= (
                        display_df[col]
                        .astype(str)
                        .str.contains(search_term, case=False, na=False)
                    )

            if mask.any():
                display_df = display_df[mask]
            else:
                st.warning("No matches found")

        st.dataframe(
            display_df.head(num_students), use_container_width=True, height=400
        )

        # Data export with filters applied
        st.write("### 💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode("utf-8")

            csv_data = convert_df_to_csv(display_df)
            st.download_button(
                "📁 Download Filtered CSV",
                csv_data,
                f"student_data_filtered_{len(display_df)}_records.csv",
                "text/csv",
                key="download-filtered-csv",
            )

        with col2:
            # Download summary statistics
            if show_summary_stats and not display_df.empty:
                numeric_cols = display_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    summary_csv = convert_df_to_csv(display_df[numeric_cols].describe())
                    st.download_button(
                        "📊 Download Summary Stats",
                        summary_csv,
                        "summary_statistics.csv",
                        "text/csv",
                        key="download-summary-csv",
                    )

        with col3:
            # Download correlation matrix
            if len(display_df.select_dtypes(include=[np.number]).columns) > 1:
                corr_csv = convert_df_to_csv(
                    display_df.select_dtypes(include=[np.number]).corr()
                )
                st.download_button(
                    "🔗 Download Correlations",
                    corr_csv,
                    "correlation_matrix.csv",
                    "text/csv",
                    key="download-corr-csv",
                )

    # summary statistics
    if show_summary_stats:
        st.write("### 📊 Summary Statistics")

        numeric_columns = filtered_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if numeric_columns:
            summary_stats = filtered_df[numeric_columns].describe()

            # Add additional statistics
            additional_stats = pd.DataFrame(
                {
                    col: {
                        "skewness": filtered_df[col].skew(),
                        "kurtosis": filtered_df[col].kurtosis(),
                        "missing": filtered_df[col].isnull().sum(),
                        "unique": filtered_df[col].nunique(),
                    }
                    for col in numeric_columns
                }
            ).T

            # Combine statistics
            full_stats = pd.concat([summary_stats.T, additional_stats], axis=1)
            st.dataframe(full_stats.round(3), use_container_width=True)

    # Data quality report
    st.write("### 🎯 Data Quality Report")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Data Completeness:**")
        missing_data = filtered_df.isnull().sum()
        completeness = (1 - missing_data / len(filtered_df)) * 100

        for col in filtered_df.columns:
            if not selected_columns or col in selected_columns:
                st.write(f"• {col}: {completeness[col]:.1f}% complete")

    with col2:
        st.write("**Data Distribution:**")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns[:5]
        for col in numeric_cols:
            if not selected_columns or col in selected_columns:
                skew = filtered_df[col].skew()
                if abs(skew) < 0.5:
                    dist_desc = "Normal"
                elif abs(skew) < 1:
                    dist_desc = "Moderate skew"
                else:
                    dist_desc = "High skew"
                st.write(f"• {col}: {dist_desc}")

    # Interactive histogram for any variable
    st.write("### 📈 Variable Distribution Explorer")

    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        hist_column = st.selectbox(
            "Select variable for detailed distribution analysis:",
            options=numeric_columns,
            index=0,
        )

        if hist_column:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    filtered_df,
                    x=hist_column,
                    color="Stress_Level",
                    marginal="box",
                    title=f"Distribution of {hist_column}",
                    color_discrete_map=color_discrete_map_stress,
                    category_orders=category_orders_stress,
                    nbins=20,
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distribution statistics
                st.write(f"**Statistics for {hist_column}:**")
                col_data = filtered_df[hist_column].dropna()

                if len(col_data) > 0:
                    st.write(f"• Mean: {col_data.mean():.3f}")
                    st.write(f"• Median: {col_data.median():.3f}")
                    st.write(f"• Std Dev: {col_data.std():.3f}")
                    st.write(f"• Min: {col_data.min():.3f}")
                    st.write(f"• Max: {col_data.max():.3f}")
                    st.write(f"• Skewness: {col_data.skew():.3f}")

                    # Outlier detection
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_data[
                        (col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)
                    ]

                    st.write(
                        f"• Outliers: {len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)"
                    )

                    # Distribution interpretation
                    st.write("### 📋 Interpretation:")
                    if abs(col_data.skew()) > 1:
                        st.write("• Highly skewed distribution")
                    elif abs(col_data.skew()) > 0.5:
                        st.write("• Moderately skewed distribution")
                    else:
                        st.write("• Approximately normal distribution")

                    if len(outliers) / len(col_data) > 0.05:
                        st.write("• High number of outliers detected")
                    elif len(outliers) > 0:
                        st.write("• Some outliers present")
                    else:
                        st.write("• No significant outliers")

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Dashboard Info**")
    st.markdown(f"• Total Students: {len(df):,}")
    st.markdown(f"• Filtered Students: {len(filtered_df):,}")
    st.markdown(f"• Features: {len(df.columns)}")

with col2:
    st.markdown("**🔧 Current Filters**")
    st.markdown(f"• Stress Levels: {', '.join(selected_stress)}")
    st.markdown(f"• GPA Range: {gpa_range[0]:.1f} - {gpa_range[1]:.1f}")
    st.markdown(f"• Balance Range: {balance_range[0]:.2f} - {balance_range[1]:.2f}")

with col3:
    st.markdown("**ℹ️ About**")
    st.markdown("• Dashboard: Student Life Balance Analytics")
    st.markdown("• Framework: Streamlit + Plotly")
    st.markdown("• Created by Marvel Pangondian, Steven Tjhia, and Muhamad Rafli Rasyiidin")

st.markdown(
    '<div style="text-align: center; color: #666; padding: 1rem;">'
    "Student Life Balance Dashboard | "
    "Data-driven insights for academic wellness"
    "</div>",
    unsafe_allow_html=True,
)
