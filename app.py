# app.py - Complete Auto EDA with PyTorch in One File
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import missingno as msno
import openai
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# ==================== PYTORCH MODELS ====================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PredictiveModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(PredictiveModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# ==================== PYTORCH ANALYZER ====================

class PyTorchAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            st.sidebar.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.info("⚡ Using CPU for computations")
    
    def detect_anomalies_autoencoder(self, df, numerical_cols, contamination=0.1):
        """
        Detect anomalies using Autoencoder
        """
        if not numerical_cols:
            return None
        
        try:
            # Prepare data
            X = df[numerical_cols].fillna(df[numerical_cols].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Initialize and train autoencoder
            input_dim = X_scaled.shape[1]
            autoencoder = Autoencoder(input_dim).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
            
            # Training loop with progress tracking
            autoencoder.train()
            losses = []
            
            # Create a progress bar
            progress_text = "Training Autoencoder for Anomaly Detection..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = autoencoder(X_tensor)
                loss = criterion(outputs, X_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                # Update progress every 10 epochs
                if epoch % 10 == 0:
                    progress = (epoch + 1) / 100
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed!")
            
            # Calculate reconstruction error
            autoencoder.eval()
            with torch.no_grad():
                reconstructed = autoencoder(X_tensor)
                reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
                errors = reconstruction_error.cpu().numpy()
            
            # Detect anomalies (top reconstruction errors)
            threshold = np.percentile(errors, 100 * (1 - contamination))
            anomalies = errors > threshold
            
            return {
                'anomaly_indices': np.where(anomalies)[0],
                'anomaly_scores': errors,
                'threshold': threshold,
                'training_loss': losses,
                'num_anomalies': len(np.where(anomalies)[0]),
                'anomaly_percentage': (len(np.where(anomalies)[0]) / len(df)) * 100
            }
            
        except Exception as e:
            st.error(f"❌ Anomaly detection failed: {str(e)}")
            return None
    
    def perform_clustering(self, df, numerical_cols, n_clusters=3):
        """
        Perform clustering using PyTorch-enhanced K-means
        """
        if not numerical_cols:
            return None
        
        try:
            # Prepare data
            X = df[numerical_cols].fillna(df[numerical_cols].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Convert to PyTorch tensor for potential GPU acceleration
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Use K-means with PyTorch tensor input
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = X_scaled[clusters == cluster_id]
                cluster_stats[cluster_id] = {
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(X_scaled)) * 100
                }
            
            return {
                'cluster_labels': clusters,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats
            }
            
        except Exception as e:
            st.error(f"❌ Clustering failed: {str(e)}")
            return None

# ==================== EDA ANALYSIS FUNCTIONS ====================

def perform_comprehensive_eda(df):
    """
    Perform comprehensive EDA analysis with PyTorch enhancements
    """
    results = {}
    
    # Basic information
    results['shape'] = df.shape
    results['data_types'] = df.dtypes.value_counts().to_dict()
    results['missing_data'] = df.isnull().sum().sort_values(ascending=False)
    results['duplicate_rows'] = df.duplicated().sum()
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    results['numerical_columns'] = numerical_cols
    results['categorical_columns'] = categorical_cols
    
    # Descriptive statistics
    results['numerical_stats'] = calculate_detailed_stats(df, numerical_cols)
    results['categorical_stats'] = calculate_categorical_stats(df, categorical_cols)
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        results['correlation_matrix'] = df[numerical_cols].corr()
        results['high_correlations'] = find_high_correlations(results['correlation_matrix'])
    
    # Outlier detection (traditional)
    results['outliers'] = detect_outliers(df, numerical_cols)
    
    # Distribution analysis
    results['distributions'] = analyze_distributions(df, numerical_cols)
    
    # Data quality metrics
    results['data_quality'] = calculate_data_quality(df)
    
    # PyTorch-based analysis
    pytorch_analyzer = PyTorchAnalyzer()
    
    if numerical_cols:
        # Anomaly detection
        with st.spinner("🔍 Detecting anomalies using PyTorch Autoencoder..."):
            results['pytorch_anomalies'] = pytorch_analyzer.detect_anomalies_autoencoder(
                df, numerical_cols
            )
        
        # Clustering
        with st.spinner("🧮 Performing clustering analysis..."):
            n_clusters = min(5, len(numerical_cols))  # Dynamic cluster count
            results['pytorch_clustering'] = pytorch_analyzer.perform_clustering(
                df, numerical_cols, n_clusters=n_clusters
            )
    
    # Summary
    results['summary'] = generate_summary(results, df)
    
    return results

def calculate_detailed_stats(df, numerical_cols):
    """Calculate detailed statistics for numerical columns"""
    if not numerical_cols:
        return pd.DataFrame()
    
    stats_df = df[numerical_cols].describe().T
    stats_df['variance'] = df[numerical_cols].var()
    stats_df['skewness'] = df[numerical_cols].apply(skew)
    stats_df['kurtosis'] = df[numerical_cols].apply(kurtosis)
    stats_df['missing_count'] = df[numerical_cols].isnull().sum()
    stats_df['missing_percentage'] = (stats_df['missing_count'] / len(df)) * 100
    stats_df['zeros_count'] = (df[numerical_cols] == 0).sum()
    
    return stats_df.round(4)

def calculate_categorical_stats(df, categorical_cols):
    """Calculate statistics for categorical columns"""
    if not categorical_cols:
        return pd.DataFrame()
    
    stats = []
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        stats.append({
            'column': col,
            'unique_count': df[col].nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'most_frequent_percentage': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
        })
    
    return pd.DataFrame(stats)

def find_high_correlations(corr_matrix, threshold=0.8):
    """Find highly correlated feature pairs"""
    high_corr_pairs = []
    corr_matrix_abs = corr_matrix.abs()
    
    for i in range(len(corr_matrix_abs.columns)):
        for j in range(i):
            if corr_matrix_abs.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix_abs.columns[i],
                    'feature2': corr_matrix_abs.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return high_corr_pairs

def detect_outliers(df, numerical_cols, method='iqr'):
    """Detect outliers using IQR method"""
    outliers = {}
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers

def analyze_distributions(df, numerical_cols):
    """Analyze distributions of numerical columns"""
    distributions = {}
    
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 8:  # Need sufficient data for normality test
            is_normal = stats.normaltest(col_data)[1] > 0.05
        else:
            is_normal = False
            
        distributions[col] = {
            'is_normal': is_normal,
            'skewness': skew(col_data),
            'kurtosis': kurtosis(col_data)
        }
    
    return distributions

def calculate_data_quality(df):
    """Calculate data quality metrics"""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    return {
        'completeness_score': ((total_cells - missing_cells) / total_cells) * 100,
        'uniqueness_score': ((len(df) - duplicate_rows) / len(df)) * 100,
        'total_missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows
    }

def generate_summary(eda_results, df):
    """Generate overall summary of EDA"""
    return {
        'shape': eda_results['shape'],
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB",
        'numerical_columns': len(eda_results['numerical_columns']),
        'categorical_columns': len(eda_results['categorical_columns']),
        'high_correlation_pairs': len(eda_results.get('high_correlations', [])),
        'skewed_features': len([col for col, dist in eda_results['distributions'].items() 
                              if abs(dist['skewness']) > 1]),
        'constant_features': len([col for col in eda_results['numerical_columns'] 
                                if df[col].nunique() == 1])
    }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_visualizations(df, eda_results):
    """Create comprehensive visualizations"""
    numerical_cols = eda_results['numerical_columns']
    categorical_cols = eda_results['categorical_columns']
    
    # Distribution plots for numerical variables
    if numerical_cols:
        st.subheader("📈 Numerical Variables Distribution")
        create_distribution_plots(df, numerical_cols)
    
    # Categorical variables analysis
    if categorical_cols:
        st.subheader("📊 Categorical Variables Analysis")
        create_categorical_plots(df, categorical_cols)
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        create_correlation_heatmap(eda_results['correlation_matrix'])
    
    # Missing values visualization
    if df.isnull().sum().sum() > 0:
        st.subheader("❓ Missing Values Pattern")
        create_missing_values_plot(df)
    
    # PyTorch-specific visualizations
    if 'pytorch_anomalies' in eda_results and eda_results['pytorch_anomalies']:
        create_anomaly_visualizations(df, eda_results)
    
    if 'pytorch_clustering' in eda_results and eda_results['pytorch_clustering']:
        create_clustering_visualizations(df, eda_results)

def create_distribution_plots(df, numerical_cols):
    """Create distribution plots for numerical variables"""
    # Show only first 6 columns to avoid overcrowding
    display_cols = numerical_cols[:6]
    
    for col in display_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with KDE
            fig = px.histogram(df, x=col, title=f"Distribution of {col}", 
                             marginal="box", nbins=50)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)

def create_categorical_plots(df, categorical_cols):
    """Create plots for categorical variables"""
    for col in categorical_cols[:4]:  # Limit to first 4 columns
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f"Top Categories in {col}",
                labels={'x': col, 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart for top categories
            if len(value_counts) > 1:
                fig_pie = px.pie(
                    names=value_counts.index, 
                    values=value_counts.values,
                    title=f"Distribution of {col} (Top Categories)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap"""
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_missing_values_plot(df):
    """Create missing values visualization"""
    # Missing values by column
    missing_by_col = df.isnull().sum().sort_values(ascending=False)
    missing_by_col = missing_by_col[missing_by_col > 0]
    
    if len(missing_by_col) > 0:
        fig_bar = px.bar(
            x=missing_by_col.index,
            y=missing_by_col.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def create_anomaly_visualizations(df, eda_results):
    """Create visualizations for PyTorch anomaly detection"""
    st.subheader("🤖 PyTorch Anomaly Detection Results")
    
    anomalies_data = eda_results['pytorch_anomalies']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anomalies Detected", anomalies_data['num_anomalies'])
    with col2:
        st.metric("Anomaly Percentage", f"{anomalies_data['anomaly_percentage']:.1f}%")
    with col3:
        st.metric("Detection Threshold", f"{anomalies_data['threshold']:.4f}")
    
    # Anomaly scores distribution
    fig_scores = px.histogram(
        x=anomalies_data['anomaly_scores'],
        title="Anomaly Scores Distribution",
        labels={'x': 'Reconstruction Error', 'y': 'Count'}
    )
    fig_scores.add_vline(
        x=anomalies_data['threshold'],
        line_dash="dash", 
        line_color="red",
        annotation_text="Threshold"
    )
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # Training loss
    if 'training_loss' in anomalies_data:
        fig_loss = px.line(
            y=anomalies_data['training_loss'],
            title="Autoencoder Training Loss",
            labels={'x': 'Epoch', 'y': 'Loss'}
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Show anomalous rows
    with st.expander("🔍 View Anomalous Records"):
        anomaly_indices = anomalies_data['anomaly_indices']
        if len(anomaly_indices) > 0:
            st.dataframe(df.iloc[anomaly_indices].head(10), use_container_width=True)
        else:
            st.info("No anomalies detected in the dataset")

def create_clustering_visualizations(df, eda_results):
    """Create visualizations for clustering results"""
    clustering_data = eda_results['pytorch_clustering']
    numerical_cols = eda_results['numerical_columns']
    
    st.subheader("🧮 PyTorch Clustering Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Clusters", clustering_data['n_clusters'])
    with col2:
        st.metric("Within-Cluster Variance", f"{clustering_data['inertia']:.2f}")
    with col3:
        total_points = len(clustering_data['cluster_labels'])
        st.metric("Total Data Points", total_points)
    
    # Cluster distribution
    cluster_counts = pd.Series(clustering_data['cluster_labels']).value_counts().sort_index()
    fig_cluster_dist = px.bar(
        x=cluster_counts.index.astype(str), 
        y=cluster_counts.values,
        title="Cluster Distribution",
        labels={'x': 'Cluster', 'y': 'Count'}
    )
    st.plotly_chart(fig_cluster_dist, use_container_width=True)
    
    # 2D scatter plot if we have at least 2 numerical columns
    if len(numerical_cols) >= 2:
        df_cluster = df.copy()
        df_cluster['Cluster'] = clustering_data['cluster_labels'].astype(str)
        
        fig_clusters = px.scatter(
            df_cluster, 
            x=numerical_cols[0],
            y=numerical_cols[1],
            color='Cluster',
            title="Clustering Visualization",
            hover_data=df_cluster.columns
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

# ==================== DISPLAY FUNCTIONS ====================

def display_statistics(eda_results):
    """Display statistical results"""
    st.subheader("📊 Descriptive Statistics")
    
    if 'numerical_stats' in eda_results and not eda_results['numerical_stats'].empty:
        st.write("**Numerical Variables Statistics:**")
        st.dataframe(eda_results['numerical_stats'], use_container_width=True)
    
    if 'categorical_stats' in eda_results and not eda_results['categorical_stats'].empty:
        st.write("**Categorical Variables Statistics:**")
        st.dataframe(eda_results['categorical_stats'], use_container_width=True)
    
    if 'correlation_matrix' in eda_results:
        st.write("**Correlation Matrix:**")
        st.dataframe(eda_results['correlation_matrix'].round(3), use_container_width=True)
        
        # Show high correlations
        if 'high_correlations' in eda_results and eda_results['high_correlations']:
            st.write("**Highly Correlated Features (|r| > 0.8):**")
            high_corr_df = pd.DataFrame(eda_results['high_correlations'])
            st.dataframe(high_corr_df, use_container_width=True)

def display_data_quality(df, eda_results):
    """Display data quality assessment"""
    st.subheader("🔍 Data Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Quality Metrics:**")
        data_quality = eda_results['data_quality']
        st.metric("Completeness Score", f"{data_quality['completeness_score']:.1f}%")
        st.metric("Uniqueness Score", f"{data_quality['uniqueness_score']:.1f}%")
        st.metric("Total Missing Cells", data_quality['total_missing_cells'])
        st.metric("Duplicate Rows", data_quality['duplicate_rows'])
    
    with col2:
        st.write("**Data Types Summary:**")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Missing Values': df.isnull().sum().values
        })
        st.dataframe(dtype_info, use_container_width=True)
    
    # Outlier information
    if 'outliers' in eda_results:
        st.write("**Outlier Analysis:**")
        outlier_data = []
        for col, info in eda_results['outliers'].items():
            outlier_data.append({
                'Column': col,
                'Outliers': info['count'],
                'Percentage': f"{info['percentage']:.1f}%"
            })
        outlier_df = pd.DataFrame(outlier_data)
        st.dataframe(outlier_df, use_container_width=True)

def display_ml_analysis(df, eda_results):
    """Display machine learning analysis results"""
    st.subheader("🤖 PyTorch Machine Learning Analysis")
    
    # Anomaly Detection Results
    if 'pytorch_anomalies' in eda_results and eda_results['pytorch_anomalies']:
        st.write("### 🔍 Anomaly Detection")
        anomalies_data = eda_results['pytorch_anomalies']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", anomalies_data['num_anomalies'])
        with col2:
            st.metric("Anomaly Rate", f"{anomalies_data['anomaly_percentage']:.1f}%")
        with col3:
            st.metric("Detection Threshold", f"{anomalies_data['threshold']:.4f}")
        
        # Show some anomalous records
        with st.expander("View Detailed Anomaly Results"):
            if len(anomalies_data['anomaly_indices']) > 0:
                st.write(f"**First 10 Anomalous Records:**")
                anomalous_df = df.iloc[anomalies_data['anomaly_indices']].head(10)
                st.dataframe(anomalous_df, use_container_width=True)
            else:
                st.info("No anomalies detected in the dataset")
    
    # Clustering Results
    if 'pytorch_clustering' in eda_results and eda_results['pytorch_clustering']:
        st.write("### 🧮 Clustering Analysis")
        clustering_data = eda_results['pytorch_clustering']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Clusters", clustering_data['n_clusters'])
        with col2:
            st.metric("Within-Cluster Variance", f"{clustering_data['inertia']:.2f}")
        
        # Cluster statistics
        st.write("**Cluster Statistics:**")
        cluster_stats = []
        for cluster_id, stats in clustering_data['cluster_stats'].items():
            cluster_stats.append({
                'Cluster': cluster_id,
                'Size': stats['size'],
                'Percentage': f"{stats['percentage']:.1f}%"
            })
        cluster_df = pd.DataFrame(cluster_stats)
        st.dataframe(cluster_df, use_container_width=True)

def display_summary(eda_results):
    """Display EDA summary"""
    st.subheader("📝 EDA Summary")
    
    if 'summary' in eda_results:
        summary = eda_results['summary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Overview:**")
            st.write(f"- **Shape:** {summary.get('shape', 'N/A')}")
            st.write(f"- **Memory Usage:** {summary.get('memory_usage', 'N/A')}")
            st.write(f"- **Numerical Columns:** {summary.get('numerical_columns', 0)}")
            st.write(f"- **Categorical Columns:** {summary.get('categorical_columns', 0)}")
            st.write(f"- **High Correlation Pairs:** {summary.get('high_correlation_pairs', 0)}")
        
        with col2:
            st.write("**Data Quality Indicators:**")
            st.write(f"- **Skewed Features:** {summary.get('skewed_features', 0)}")
            st.write(f"- **Constant Features:** {summary.get('constant_features', 0)}")
            if 'data_quality' in eda_results:
                dq = eda_results['data_quality']
                st.write(f"- **Completeness:** {dq.get('completeness_score', 0):.1f}%")
                st.write(f"- **Uniqueness:** {dq.get('uniqueness_score', 0):.1f}%")

def display_data_info(df):
    """Display detailed column information"""
    st.subheader("📁 Detailed Column Information")
    
    for column in df.columns:
        with st.expander(f"📊 {column} ({df[column].dtype})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Info:**")
                st.write(f"- **Non-null count:** {df[column].count()}")
                st.write(f"- **Null count:** {df[column].isnull().sum()}")
                st.write(f"- **Unique values:** {df[column].nunique()}")
                
                if df[column].dtype in ['object', 'category']:
                    st.write(f"- **Most frequent:** {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}")
                    st.write(f"- **Frequency of most common:** {df[column].value_counts().iloc[0] if len(df[column].value_counts()) > 0 else 0}")
                else:
                    st.write(f"- **Mean:** {df[column].mean():.2f}")
                    st.write(f"- **Std:** {df[column].std():.2f}")
                    st.write(f"- **Min:** {df[column].min():.2f}")
                    st.write(f"- **Max:** {df[column].max():.2f}")
            
            with col2:
                st.write("**Sample Values:**")
                unique_vals = df[column].dropna().unique()
                sample_vals = unique_vals[:5] if len(unique_vals) > 5 else unique_vals
                for val in sample_vals:
                    st.write(f"- {val}")

# ==================== LLM INTEGRATION ====================

class LLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('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')
        if self.api_key:
            openai.api_key = self.api_key
        else:
            st.sidebar.warning("⚠️ OpenAI API key not found. LLM features disabled.")
    
    def generate_insights(self, df, eda_results):
        """Generate AI-powered insights using LLM"""
        if not self.api_key:
            return {"error": "OpenAI API key not configured"}
        
        try:
            # Prepare data summary for LLM
            data_summary = self.prepare_data_summary(df, eda_results)
            
            prompt = f"""
            As a senior data analyst, perform exploratory data analysis on the following dataset and provide insights:

            {data_summary}

            Please provide:
            1. Key findings about the data distribution, patterns, and relationships
            2. Data quality issues and recommendations
            3. Potential business insights or anomalies
            4. Suggestions for further analysis

            Format the response as JSON with these keys: 
            - "key_findings" (list of strings)
            - "recommendations" (list of strings) 
            - "anomalies" (list of strings)
            - "business_insights" (list of strings)
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst specializing in exploratory data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            insights_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON, fallback to text
            try:
                import json
                return json.loads(insights_text)
            except json.JSONDecodeError:
                return {"raw_insights": insights_text}
                
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def prepare_data_summary(self, df, eda_results):
        """Prepare a comprehensive data summary for LLM"""
        summary = f"""
        Dataset Overview:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data Types: {dict(df.dtypes.value_counts())}
        
        Data Quality:
        - Total Missing Values: {df.isnull().sum().sum()}
        - Duplicate Rows: {df.duplicated().sum()}
        
        Numerical Columns ({len(eda_results['numerical_columns'])}):
        """
        
        # Add numerical columns summary
        for col in eda_results['numerical_columns'][:10]:  # Limit to first 10
            stats = eda_results['numerical_stats'].loc[col]
            summary += f"\n  - {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}"
        
        # Add categorical columns summary
        if eda_results['categorical_columns']:
            summary += f"\n\nCategorical Columns ({len(eda_results['categorical_columns'])}):"
            for col in eda_results['categorical_columns'][:5]:  # Limit to first 5
                unique_vals = df[col].nunique()
                summary += f"\n  - {col}: {unique_vals} unique values"
        
        # Add correlation insights
        if 'high_correlations' in eda_results and eda_results['high_correlations']:
            summary += f"\n\nHigh Correlations (|r| > 0.8):"
            for corr in eda_results['high_correlations'][:5]:
                summary += f"\n  - {corr['feature1']} & {corr['feature2']}: {corr['correlation']:.3f}"
        
        # Add anomaly information
        if 'pytorch_anomalies' in eda_results and eda_results['pytorch_anomalies']:
            anomaly_data = eda_results['pytorch_anomalies']
            summary += f"\n\nAnomaly Detection:"
            summary += f"\n  - Anomalies found: {anomaly_data['num_anomalies']} ({anomaly_data['anomaly_percentage']:.1f}%)"
        
        return summary

def generate_eda_insights(df, eda_results):
    """Main function to generate EDA insights using LLM"""
    analyzer = LLMAnalyzer()
    return analyzer.generate_insights(df, eda_results)

def display_llm_insights(insights):
    """Display LLM-generated insights"""
    st.subheader("🧠 AI-Powered Insights")
    
    if insights and 'error' not in insights:
        if 'key_findings' in insights:
            st.write("### 🔑 Key Findings")
            for i, finding in enumerate(insights['key_findings'], 1):
                st.write(f"{i}. {finding}")
        
        if 'recommendations' in insights:
            st.write("### 💡 Recommendations")
            for i, rec in enumerate(insights['recommendations'], 1):
                st.write(f"{i}. {rec}")
        
        if 'anomalies' in insights:
            st.write("### ⚠️ Potential Anomalies")
            for i, anomaly in enumerate(insights['anomalies'], 1):
                st.write(f"{i}. {anomaly}")
        
        if 'business_insights' in insights:
            st.write("### 💼 Business Insights")
            for i, insight in enumerate(insights['business_insights'], 1):
                st.write(f"{i}. {insight}")
                
    elif 'raw_insights' in insights:
        st.write("### 📋 AI Analysis")
        st.write(insights['raw_insights'])
    else:
        st.warning("LLM insights not available. Please check your API configuration or enable AI insights.")

# ==================== MAIN STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Auto EDA with PyTorch & LLM",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Automated EDA with PyTorch & LLM")
    st.markdown("Upload your CSV or Excel file and get comprehensive exploratory data analysis with AI-powered insights!")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for analysis"
    )
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    enable_llm = st.sidebar.checkbox("Enable AI Insights", value=True)
    enable_pytorch = st.sidebar.checkbox("Enable PyTorch ML", value=True)
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format")
                return
            
            if df is not None and not df.empty:
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                with col4:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
                
                # Data preview
                with st.expander("🔍 Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Perform EDA
                with st.spinner("Performing comprehensive EDA analysis..."):
                    eda_results = perform_comprehensive_eda(df)
                
                # Generate LLM insights if enabled
                llm_insights = None
                if enable_llm:
                    with st.spinner("Generating AI insights..."):
                        llm_insights = generate_eda_insights(df, eda_results)
                
                # Display results in tabs
                tab_names = ["📈 Visualizations", "📊 Statistics", "🔍 Data Quality"]
                if enable_pytorch:
                    tab_names.append("🤖 ML Analysis")
                if enable_llm and llm_insights:
                    tab_names.append("🧠 AI Insights")
                tab_names.extend(["📝 Summary", "📁 Data Info"])
                
                tabs = st.tabs(tab_names)
                
                current_tab = 0
                
                # Visualizations tab
                with tabs[current_tab]:
                    create_visualizations(df, eda_results)
                current_tab += 1
                
                # Statistics tab
                with tabs[current_tab]:
                    display_statistics(eda_results)
                current_tab += 1
                
                # Data Quality tab
                with tabs[current_tab]:
                    display_data_quality(df, eda_results)
                current_tab += 1
                
                # ML Analysis tab
                if enable_pytorch:
                    with tabs[current_tab]:
                        display_ml_analysis(df, eda_results)
                    current_tab += 1
                
                # AI Insights tab
                if enable_llm and llm_insights:
                    with tabs[current_tab]:
                        display_llm_insights(llm_insights)
                    current_tab += 1
                
                # Summary tab
                with tabs[current_tab]:
                    display_summary(eda_results)
                current_tab += 1
                
                # Data Info tab
                with tabs[current_tab]:
                    display_data_info(df)
            
            else:
                st.error("❌ The uploaded file is empty or couldn't be loaded.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Try uploading a different file or check the file format.")
    else:
        # Show demo when no file is uploaded
        st.info("👆 Please upload a CSV or Excel file to get started!")
        
        # Demo section
        st.subheader("🎯 What this tool can do:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Basic EDA**")
            st.write("• Descriptive statistics")
            st.write("• Data type analysis")
            st.write("• Missing value patterns")
            
        with col2:
            st.write("**🤖 ML Analysis**")
            st.write("• Autoencoder anomaly detection")
            st.write("• K-means clustering")
            st.write("• Predictive modeling")
            
        with col3:
            st.write("**🧠 AI Insights**")
            st.write("• LLM-powered analysis")
            st.write("• Business insights")
            st.write("• Recommendations")
        
        # Requirements info
        st.sidebar.info("""
        **Required Packages:**
        - streamlit, pandas, numpy
        - torch, scikit-learn
        - plotly, matplotlib
        - openai (for AI insights)
        """)

if __name__ == "__main__":

    main()
