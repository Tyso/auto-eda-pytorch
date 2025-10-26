# app.py - With LLM AI Insights
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import skew, kurtosis
import openai
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Auto EDA with AI Insights",
    page_icon="📊",
    layout="wide"
)

# ==================== LLM ANALYZER ====================

class LLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('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')
        if self.api_key:
            openai.api_key = self.api_key
            st.sidebar.success("✅ OpenAI API configured")
        else:
            st.sidebar.warning("⚠️ OpenAI API key not found")
    
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
        for col in eda_results['numerical_columns'][:10]:
            stats = eda_results['numerical_stats'].loc[col]
            summary += f"\n  - {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}"
        
        # Add categorical columns summary
        if eda_results['categorical_columns']:
            summary += f"\n\nCategorical Columns ({len(eda_results['categorical_columns'])}):"
            for col in eda_results['categorical_columns'][:5]:
                unique_vals = df[col].nunique()
                summary += f"\n  - {col}: {unique_vals} unique values"
        
        # Add correlation insights
        if 'correlation_matrix' in eda_results:
            corr_matrix = eda_results['correlation_matrix']
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                summary += f"\n\nHigh Correlations (|r| > 0.8):"
                for corr in high_corr_pairs[:3]:
                    summary += f"\n  - {corr['feature1']} & {corr['feature2']}: {corr['correlation']:.3f}"
        
        return summary

def generate_eda_insights(df, eda_results):
    """Main function to generate EDA insights using LLM"""
    analyzer = LLMAnalyzer()
    return analyzer.generate_insights(df, eda_results)

def display_llm_insights(insights):
    """Display LLM-generated insights"""
    st.header("🧠 AI-Powered Insights")
    
    if insights and 'error' not in insights:
        if 'key_findings' in insights:
            st.subheader("🔑 Key Findings")
            for i, finding in enumerate(insights['key_findings'], 1):
                st.write(f"{i}. {finding}")
        
        if 'recommendations' in insights:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(insights['recommendations'], 1):
                st.write(f"{i}. {rec}")
        
        if 'anomalies' in insights:
            st.subheader("⚠️ Potential Anomalies")
            for i, anomaly in enumerate(insights['anomalies'], 1):
                st.write(f"{i}. {anomaly}")
        
        if 'business_insights' in insights:
            st.subheader("💼 Business Insights")
            for i, insight in enumerate(insights['business_insights'], 1):
                st.write(f"{i}. {insight}")
                
    elif 'raw_insights' in insights:
        st.subheader("📋 AI Analysis")
        st.write(insights['raw_insights'])
    else:
        st.warning("LLM insights not available. Please check your OpenAI API configuration.")

# ==================== MAIN EDA FUNCTIONS ====================

def main():
    st.title("📊 Auto EDA with AI Insights")
    st.markdown("Upload your CSV file to get comprehensive data analysis with AI-powered insights!")
    
    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    
    # LLM toggle
    enable_llm = st.sidebar.checkbox("Enable AI Insights", value=True)
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.error("The uploaded file is empty.")
                return
            
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
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
            
            # Data preview
            with st.expander("🔍 Data Preview (First 10 rows)"):
                st.dataframe(df.head(10))
            
            # Perform EDA
            with st.spinner("Analyzing your data..."):
                eda_results = perform_eda(df)
            
            # Generate LLM insights if enabled
            llm_insights = None
            if enable_llm:
                with st.spinner("Generating AI insights..."):
                    llm_insights = generate_eda_insights(df, eda_results)
            
            # Display results in tabs
            tab_names = ["📈 Statistics", "📊 Visualizations", "🔍 Data Quality", "📁 Data Info"]
            if enable_llm and llm_insights:
                tab_names.append("🧠 AI Insights")
            
            tabs = st.tabs(tab_names)
            
            current_tab = 0
            
            with tabs[current_tab]:
                show_statistics(eda_results, df)
            current_tab += 1
            
            with tabs[current_tab]:
                show_visualizations(df, eda_results)
            current_tab += 1
            
            with tabs[current_tab]:
                show_data_quality(df, eda_results)
            current_tab += 1
            
            with tabs[current_tab]:
                show_data_info(df)
            current_tab += 1
            
            # AI Insights tab
            if enable_llm and llm_insights:
                with tabs[current_tab]:
                    display_llm_insights(llm_insights)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started!")
        
        # API key instructions
        with st.expander("🔑 How to set up AI Insights"):
            st.markdown("""
            ### To enable AI-powered insights:
            1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Add it to your Streamlit Cloud secrets:
               - Go to your app settings in Streamlit Community Cloud
               - Click on "Secrets"
               - Add: `OPENAI_API_KEY = "your-api-key-here"`
            
            ### Features included:
            - **AI Analysis**: GPT-powered data insights
            - **Key Findings**: Automatic pattern detection
            - **Recommendations**: Data quality suggestions
            - **Business Insights**: Actionable recommendations
            """)

def perform_eda(df):
    """Perform comprehensive EDA analysis"""
    results = {}
    
    # Basic information
    results['shape'] = df.shape
    results['data_types'] = df.dtypes.value_counts().to_dict()
    results['missing_data'] = df.isnull().sum().sort_values(ascending=False)
    results['duplicate_rows'] = df.duplicated().sum()
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    results['numerical_columns'] = numerical_cols
    results['categorical_columns'] = categorical_cols
    
    # Descriptive statistics
    if numerical_cols:
        stats_df = df[numerical_cols].describe().T
        stats_df['variance'] = df[numerical_cols].var()
        stats_df['skewness'] = df[numerical_cols].apply(skew)
        stats_df['kurtosis'] = df[numerical_cols].apply(kurtosis)
        stats_df['missing_count'] = df[numerical_cols].isnull().sum()
        stats_df['missing_percentage'] = (stats_df['missing_count'] / len(df)) * 100
        results['numerical_stats'] = stats_df.round(4)
    
    # Categorical statistics
    if categorical_cols:
        cat_stats = []
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            cat_stats.append({
                'column': col,
                'unique_count': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            })
        results['categorical_stats'] = pd.DataFrame(cat_stats)
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        results['correlation_matrix'] = df[numerical_cols].corr()
    
    # Data quality metrics
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    results['data_quality'] = {
        'completeness_score': ((total_cells - missing_cells) / total_cells) * 100,
        'uniqueness_score': ((len(df) - duplicate_rows) / len(df)) * 100,
        'total_missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows
    }
    
    return results

def show_statistics(eda_results, df):
    """Display statistical results"""
    st.header("Descriptive Statistics")
    
    if 'numerical_stats' in eda_results:
        st.subheader("Numerical Variables Statistics")
        st.dataframe(eda_results['numerical_stats'])
    
    if 'categorical_stats' in eda_results:
        st.subheader("Categorical Variables Statistics")
        st.dataframe(eda_results['categorical_stats'])
    
    if 'correlation_matrix' in eda_results:
        st.subheader("Correlation Matrix")
        fig = px.imshow(
            eda_results['correlation_matrix'],
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df, eda_results):
    """Create comprehensive visualizations"""
    numerical_cols = eda_results['numerical_columns']
    categorical_cols = eda_results['categorical_columns']
    
    # Distribution plots for numerical variables
    if numerical_cols:
        st.header("📈 Numerical Variables Distribution")
        
        # Show first 4 numerical columns
        for col in numerical_cols[:4]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=50)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Categorical variables analysis
    if categorical_cols:
        st.header("📊 Categorical Variables Analysis")
        
        # Show first 3 categorical columns
        for col in categorical_cols[:3]:
            value_counts = df[col].value_counts().head(10)
            
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

def show_data_quality(df, eda_results):
    """Display data quality assessment"""
    st.header("🔍 Data Quality Report")
    
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
    
    # Missing values visualization
    if df.isnull().sum().sum() > 0:
        st.subheader("Missing Values Analysis")
        missing_by_col = df.isnull().sum().sort_values(ascending=False)
        missing_by_col = missing_by_col[missing_by_col > 0]
        
        if len(missing_by_col) > 0:
            fig = px.bar(
                x=missing_by_col.index,
                y=missing_by_col.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

def show_data_info(df):
    """Display detailed data information"""
    st.header("📁 Detailed Column Information")
    
    for column in df.columns:
        with st.expander(f"{column} ({df[column].dtype})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"- Non-null count: {df[column].count()}")
                st.write(f"- Null count: {df[column].isnull().sum()}")
                st.write(f"- Unique values: {df[column].nunique()}")
                
                if df[column].dtype in ['object']:
                    st.write(f"- Most frequent: {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}")
                else:
                    st.write(f"- Mean: {df[column].mean():.2f}")
                    st.write(f"- Standard deviation: {df[column].std():.2f}")
            
            with col2:
                st.write("**Sample Values:**")
                sample_values = df[column].dropna().unique()[:5]
                for val in sample_values:
                    st.write(f"- {val}")

if __name__ == "__main__":
    main()
