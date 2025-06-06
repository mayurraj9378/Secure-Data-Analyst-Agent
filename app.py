import json
import tempfile
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
from groq import Groq
import duckdb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Secure Data Analyst Agent",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Secure API Key Management ---
def get_secret(key_name, input_label):
    """Load API key from environment variables or session state"""
    try:
        # 1. Try environment variables
        env_key = os.getenv(key_name)
        if env_key:
            return env_key
        
        # 2. Fallback to session state (from user input)
        if key_name in st.session_state:
            return st.session_state[key_name]
        return None
    
    except Exception as e:
        st.error(f"Error loading {key_name}: {str(e)}")
        return None

def preprocess_and_save(uploaded_file):
    """File processing with error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format")
        
        # Clean column names
        df.columns = [col.lower().replace(' ', '_').replace(r'[^\w]', '') 
                     for col in df.columns]
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name, list(df.columns), df
    
    except Exception as e:
        st.error(f"File processing failed: {str(e)}")
        return None, None, None

def run_sql_query(df, query):
    """Execute SQL query using duckdb"""
    try:
        # Create an in-memory DuckDB database
        con = duckdb.connect()
        con.register("uploaded_data", df)
        result = con.execute(query).fetchdf()
        return result.to_markdown()
    except Exception as e:
        return f"SQL Query Error: {str(e)}"

class DeepSeekAnalystAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-reasoner"
    
    def analyze_data(self, df: pd.DataFrame, query: str) -> str:
        """Enhanced analysis with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": f"Analyze this data:\n{df.head().to_markdown()}\n\nQuestion: {query}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis failed: {str(e)}"

def main():
    st.title("🔒 Secure Data Analyst Agent")
    
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("API Configuration")
        st.info("Enter your API keys below or set them in a .env file.")
        
        # Input fields for API keys
        groq_key_input = st.text_input("Groq API Key:", type="password", key="groq_key_input")
        deepseek_key_input = st.text_input("DeepSeek API Key:", type="password", key="deepseek_key_input")
        
        # Load keys from env or input
        groq_key = get_secret("GROQ_API_KEY", "Groq API Key:") or groq_key_input
        deepseek_key = get_secret("DEEPSEEK_API_KEY", "DeepSeek API Key:") or deepseek_key_input
        
        if groq_key:
            st.session_state.GROQ_API_KEY = groq_key
            st.success("✅ Groq key loaded")
        else:
            st.warning("⚠️ Groq API key required for SQL Query mode")
        
        if deepseek_key:
            st.session_state.DEEPSEEK_API_KEY = deepseek_key
            st.success("✅ DeepSeek key loaded")
        else:
            st.warning("⚠️ DeepSeek API key required for AI Insights mode")
        
        st.divider()
        analysis_mode = st.selectbox(
            "Analysis Mode:",
            ["SQL Query", "AI Insights (DeepSeek)", "Both"]
        )

    # --- Main Interface ---
    uploaded_file = st.file_uploader("📁 Upload Data", type=["csv", "xlsx"])
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if df is not None:
            st.session_state.update({
                "df": df,
                "temp_path": temp_path,
                "columns": columns
            })
            
            # Data Preview
            with st.expander("🔍 Data Preview"):
                st.dataframe(df.head())
                st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            
            # Visualization
            st.subheader("📊 Visualization")
            col1, col2 = st.columns(2)
            with col1:
                viz_type = st.selectbox(
                    "Chart Type",
                    ["Histogram", "Scatter Plot", "Box Plot", "Bar Chart"]
                )
            with col2:
                selected_cols = st.multiselect("Select Columns", columns)
            
            if st.button("Generate") and selected_cols:
                fig = None
                if viz_type == "Histogram":
                    fig = px.histogram(df, x=selected_cols[0])
                elif viz_type == "Scatter Plot" and len(selected_cols) > 1:
                    fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1])
                elif viz_type == "Box Plot":
                    fig = px.box(df, y=selected_cols[0])
                elif viz_type == "Bar Chart":
                    fig = px.bar(df[selected_cols[0]].value_counts().head(10))
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            st.subheader("🤖 Data Analysis")
            query = st.text_area("Enter your question:")
            
            if st.button("Analyze") and query:
                if analysis_mode in ["SQL Query", "Both"] and "GROQ_API_KEY" in st.session_state:
                    try:
                        # Use duckdb directly for SQL queries
                        st.markdown(run_sql_query(df, query))
                    except Exception as e:
                        st.error(f"SQL Analysis Error: {str(e)}")
                
                if analysis_mode in ["AI Insights (DeepSeek)", "Both"] and "DEEPSEEK_API_KEY" in st.session_state:
                    try:
                        agent = DeepSeekAnalystAgent(st.session_state.DEEPSEEK_API_KEY)
                        st.markdown(agent.analyze_data(df, query))
                    except Exception as e:
                        st.error(f"AI Analysis Error: {str(e)}")

if __name__ == "__main__":
    main()
    