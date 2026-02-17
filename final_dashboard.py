import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import base64
from datetime import datetime
import io
from streamlit_echarts import st_echarts
import forecasting_engine as fe
import time
import uuid
import numpy as np
from sklearn.metrics import mean_squared_error

# --- CONFIG ---
# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Spare Parts Planning Solution", initial_sidebar_state="collapsed")

# --- ASSETS ---
TECHM_LOGO = "pictures_dns/Tech_Mahindra-Logo.wine_.png"
IIMU_LOGO = "pictures_dns/IIMU-Logo-1080px-02.webp"

# --- SKU METADATA (From Reports) ---
SKU_META = {
    'PD2976': {'Name': 'Transmission Fluid (Standard)', 'LeadTime': 41.45},
    'PD457':  {'Name': 'Engine Oil (Premium)', 'LeadTime': 14.25},
    'PD1399': {'Name': 'Suspension Shocks', 'LeadTime': 28.31},
    'PD3978': {'Name': 'Radiator/Cooling', 'LeadTime': 16.38},
    'PD238':  {'Name': 'Transmission Fluid (Premium)', 'LeadTime': 75.04},
    'PD7820': {'Name': 'Radiator Hose', 'LeadTime': 35.18},
    'PD391':  {'Name': 'Brake Pads', 'LeadTime': 3.17},
    'PD112':  {'Name': 'Engine Filters', 'LeadTime': 4.33},
    'PD293':  {'Name': 'Wiper Blades', 'LeadTime': 8.50},
    'PD2782': {'Name': 'Interior Trim', 'LeadTime': 9.00},
    'PD2801': {'Name': 'Gasket Kits', 'LeadTime': 10.56}
}

# --- DATA LOADING ---
@st.cache_data
def load_db():
    if os.path.exists('Dashboard_Database.csv'):
        return pd.read_csv('Dashboard_Database.csv')
    return pd.DataFrame()

@st.cache_data
def load_future():
    if os.path.exists('Future_Forecast_Database.csv'):
        df = pd.read_csv('Future_Forecast_Database.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()

def get_history_df():
    if os.path.exists('history.csv'):
        df = pd.read_csv('history.csv')
        # Robust cleanup: Drop rows where Part is NaN or 'SUMMARY'
        df = df.dropna(subset=['Part'])
        df = df[~df['Part'].astype(str).str.contains("SUMMARY")]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    return pd.DataFrame()

# --- HELPER: EXCEL DOWNLOAD ---
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- PAGES ---

def get_base64_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def landing_page():
    # Load Font
    font_path = "pictures_dns/Canela-Regular-Trial.otf"
    font_css = ""
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            font_b64 = base64.b64encode(f.read()).decode()
            font_css = f"""
            @font-face {{
                font-family: 'Canela';
                src: url('data:font/otf;base64,{font_b64}') format('opentype');
            }}
            """

    # Video Background
    video_path = "pictures_dns/intro3.mp4"
    video_base64 = ""
    if os.path.exists(video_path):
        video_base64 = get_base64_video(video_path)
    
    # CSS for Fullscreen & Buttons
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');
        
        {font_css}
        
        /* Global Reset for Landing Only */
        .stApp {{
            background-color: transparent !important;
        }}
        
        .main .block-container {{
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }}
        
        /* Video Background */
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
            opacity: 0.45; /* Increased transparency */
        }}
        
        /* Overlay Container */
        .landing-container {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 1;
            width: 100%;
        }}
        
        /* Texts */
        .landing-subtitle {{
            font-family: 'Playfair Display', 'Canela', serif;
            font-size: 2vw;
            color: #333333; /* Dark Grey */
            margin-bottom: 0;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        
        /* Global Font Override */
        html, body, [class*="css"], font, div, p, span, h1, h2, h3, h4, h5, h6, .stMarkdown, .stButton button, .stSelectbox, .stNumberInput, * {{
            font-family: 'Canela', 'Playfair Display', serif !important;
        }}
        
        /* EXCEPTION: Home Buttons (Default Font) */
        div.stButton > button[kind="primary"], div.stButton > button[kind="secondary"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol" !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}

        .landing-title {{
            font-family: 'Playfair Display', 'Canela', serif;
            font-size: 2.5vw; /* Reduced size further */
            color: #000000;
            margin-top: 0;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 12px 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-shadow: none;
            line-height: 1.2;
            font-weight: 700;
            white-space: nowrap;
        }}
        
        /* Buttons Custom Styles */
        div.stButton > button[kind="primary"] {{
            background: #D32F2F;
            border: 2px solid #000000; /* Black Border */
            color: #FFFFFF !important; /* White Text */
            font-size: 1.2rem;
            padding: 0.75rem 2rem;
            border-radius: 30px;
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
            box-shadow: 0 0 20px rgba(211, 47, 47, 0.6), 0 0 10px rgba(211, 47, 47, 0.4);
            font-weight: 600;
        }}
        div.stButton > button[kind="primary"]:hover {{
            box-shadow: 0 0 35px rgba(211, 47, 47, 0.8), 0 0 60px rgba(211, 47, 47, 0.6); /* Super Glow */
            border-color: #FF5252;
            transform: scale(1.05);
            color: #FFFFFF !important;
            background: #E53935;
        }}
        
        div.stButton > button[kind="secondary"] {{
            background: #FDD835; /* Yellow 600 */
            border: 2px solid #000000; /* Black Border */
            color: #000000 !important; /* Black Text */
            font-size: 1.25rem;
            padding: 1rem 2.5rem;
            border-radius: 30px;
            font-weight: 600;
            box-shadow: 0 0 20px rgba(253, 216, 53, 0.6);
            transition: all 0.3s ease;
        }}
        div.stButton > button[kind="secondary"]:hover {{
            background: #FFEE58;
            transform: scale(1.05);
            box-shadow: 0 0 35px rgba(253, 216, 53, 0.8), 0 0 60px rgba(253, 216, 53, 0.6); /* Super Glow */
            color: #000000 !important; /* Keep Text Black */
            border-color: #000000; /* Keep Border Black */
        }}

    </style>
    
    <!-- Video Element -->
    <video autoplay loop muted playsinline id="myVideo">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    
    <!-- Overlay Content -->
    <div class="landing-container">
        <h1 class="landing-title">Automobile Spare Parts Forecasting Solution</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Button Logic - Center alignment
    st.markdown("<div style='height: 45vh'></div>", unsafe_allow_html=True)
    
    # Corrected Ratios: Spacers=1, Buttons=2 (Wider) to prevent stacking
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    
    with c2:
        # Dashboard Button
        if st.button("OPEN DASHBOARD", type="primary", use_container_width=True):
            st.session_state['page'] = 'dashboard'
            st.rerun()
            
    with c3:
        # Forecasting Tool
        if st.button("FORECASTING TOOL", type="secondary", use_container_width=True):
            st.session_state['page'] = 'tool'
            st.rerun()



def generate_seasonal_lead_time(forecast_series, base_lt):
    # Heuristic: Lead times increase during high demand (Oct-Dec) and Monsoon (July)
    # Norm: Lead(m) = Base * (1 + Factor).
    # Since we don't have true seasonality data, we use the demand volume itself as a proxy for supply chain stress.
    # Scaled 0 to 1 based on prediction range.
    if forecast_series.empty:
        return []
    
    vals = forecast_series.values
    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        norm = np.zeros(len(vals))
    else:
        norm = (vals - min_v) / (max_v - min_v)
        
    # Scale factor: Max 20% increase in lead time for peak demand
    seasonal_lt = base_lt * (1 + 0.2 * norm)
    return seasonal_lt

def get_best_model_info(db_df, part, loc):
    # Filter for SKU/Loc
    subset = db_df[(db_df['Part'] == part) & (db_df['Location'] == loc) & (db_df['Model'] != 'Actual')]
    if subset.empty:
        return None
    
def get_best_model_info(db_df, part, loc):
    # Filter for SKU/Loc
    subset = db_df[(db_df['Part'] == part) & (db_df['Location'] == loc)]
    
    if subset.empty:
        return None
        
    # Separate Actuals and Models
    actuals = subset[subset['Model'] == 'Actual'][['Split', 'Date', 'Value']].rename(columns={'Value': 'Actual'})
    models_df = subset[subset['Model'] != 'Actual']
    
    if models_df.empty or actuals.empty:
        return None
        
    # Merge to calculate Bias if missing
    # We need to calculate metrics per (Split, Model)
    # Group by Split, Model
    
    leaderboard = []
    
    for (split, model), group in models_df.groupby(['Split', 'Model']):
        # Get corresponding actuals for this split
        split_actuals = actuals[actuals['Split'] == split]
        
        # Merge on Date
        merged = pd.merge(group, split_actuals, on=['Date'], how='inner')
        
        if merged.empty:
            continue
            
        # Calculate Metrics
        # RMSE
        rmse = np.sqrt(mean_squared_error(merged['Actual'], merged['Value']))
        
        # MAPE
        # Handle division by zero
        actual_safe = merged['Actual'].replace(0, 0.001)
        mape = np.mean(np.abs((merged['Actual'] - merged['Value']) / actual_safe))
        
        # Bias (Mean Error) -> Forecast - Actual
        bias = np.mean(merged['Value'] - merged['Actual'])
        
        leaderboard.append({
            'Split': split,
            'Model': model,
            'RMSE': rmse,
            'MAPE': mape,
            'Bias': bias, # Keep raw bias for display
            'AbsBias': abs(bias) # Use absolute for scoring magnitude
        })
        
    if not leaderboard:
        return None
        
    lb_df = pd.DataFrame(leaderboard)
    
    # --- COMPOSITE SCORE CALCULATION ---
    # Normalize Metrics (Lower is better for all)
    # Score = 0.4*Norm_MAPE + 0.4*Norm_RMSE + 0.2*Norm_Bias
    
    def normalize(series):
        if series.max() == series.min():
            return np.zeros(len(series))
        return (series - series.min()) / (series.max() - series.min())
        
    lb_df['Norm_RMSE'] = normalize(lb_df['RMSE'])
    lb_df['Norm_MAPE'] = normalize(lb_df['MAPE'])
    lb_df['Norm_Bias'] = normalize(lb_df['AbsBias']) # Normalize Absolute Bias
    
    lb_df['Score'] = (0.4 * lb_df['Norm_RMSE']) + (0.4 * lb_df['Norm_MAPE']) + (0.2 * lb_df['Norm_Bias'])
    
    # Find Best
    best_row = lb_df.loc[lb_df['Score'].idxmin()]
    
    return best_row

def render_navbar():
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Home", use_container_width=True):
            st.session_state['page'] = 'landing'
            st.rerun()
    with c2:
        if st.button("Dashboard", use_container_width=True):
            st.session_state['page'] = 'dashboard'
            st.rerun()
    with c3:
        if st.button("Forecasting Tool", use_container_width=True):
            st.session_state['page'] = 'tool'
            st.rerun()
    with c4:
        if st.button("About", use_container_width=True):
            st.session_state['page'] = 'about'
            st.rerun()
    st.divider()

def dashboard_page():
    render_navbar()
    # --- HEADER ---
    st.markdown("""
    <style>
    /* Hover Glow for Download Button */
    .stDownloadButton > button:hover {
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.8), 0 0 30px rgba(46, 204, 113, 0.4) !important;
        border-color: #2ecc71 !important;
        color: #000000 !important;
        background-color: #ffffff !important;
        transition: all 0.3s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### AI/ML Spare Parts Planning Solution")
    st.divider()

    df = load_db()
    future_df = load_future()
    hist_df = get_history_df()
    
    # --- 1. SELECTION ---
    # Use simple columns to ensure side-by-side (50/50 split)
    if not df.empty:
        parts = df['Part'].unique()
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            sel_part = st.selectbox("Select SKU", parts)
        with c_sel2:
            locs = df[df['Part'] == sel_part]['Location'].unique()
            sel_loc = st.selectbox("Select Location", locs)
    else:
        st.error("Data not found.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 2. DETAILS HEADER ---
    meta = SKU_META.get(sel_part, {'Name': 'Unknown Part', 'LeadTime': 0})
    st.markdown(f"<h1 style='text-align: center; color: #2c3e50; font-family: Canela;'>{sel_part} - {meta['Name']}</h1>", unsafe_allow_html=True)
    st.divider()

    # --- LOGIC: BEST MODEL SELECTION ---
    best_info = get_best_model_info(df, sel_part, sel_loc)
    
    if best_info is None:
        st.warning("Not enough data to determine best model.")
        best_model_name = "Weighted Ensemble" # Default
        best_split_name = "Split 3.5y/0.5y"
    else:
        best_model_name = best_info['Model']
        best_split_name = best_info['Split']
        best_mape = best_info['MAPE']
        best_rmse = best_info['RMSE']
        best_bias = best_info.get('Bias', 0)

    # --- 3. GRAPH 1: 2025 FORECAST (EXECUTIVE VIEW) ---
    st.markdown(f"<h2 style='text-align: center; font-family: Canela;'>2025 Forecast</h2>", unsafe_allow_html=True)
    
    # Fetch Future Data for Best Model
    f_sub = future_df[(future_df['Part'] == sel_part) & 
                      (future_df['Location'] == sel_loc) & 
                      (future_df['Model'] == best_model_name)]
    
    # Fallback logic if specific model missing
    if f_sub.empty:
        f_sub = future_df[(future_df['Part'] == sel_part) & (future_df['Location'] == sel_loc)]
        if not f_sub.empty:
            f_sub = f_sub[f_sub['Model'] == f_sub['Model'].iloc[0]] # Take first available
    
    if not f_sub.empty:
        f_sub = f_sub.sort_values('Date')
        
        # Calculate Variable Lead Time
        seasonal_lead_times = generate_seasonal_lead_time(f_sub['Value'], meta['LeadTime'])
        f_sub['LeadTime_Var'] = seasonal_lead_times
        
        # Metrics for Tiles
        total_demand = f_sub['Value'].sum()
        avg_lead_time_year = f_sub['LeadTime_Var'].mean()
        
        # Big Rounded Rectangles
        xm1, xm2 = st.columns(2)
        with xm1:
            st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 10px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #2e7d32; margin:0; font-family: Canela; font-size: 1.2em;">Yearly Demand Forecast</h3>
                <h1 style="color: #1b5e20; margin:0; font-size: 2em; font-family: Canela;">{total_demand:,.0f}</h1>
                <p style="color: #4caf50; margin:0; font-size: 0.9em;">Units (2025)</p>
            </div>
            """, unsafe_allow_html=True)
        with xm2:
            st.markdown(f"""
            <div style="background-color: #fff3e0; padding: 10px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #ef6c00; margin:0; font-family: Canela; font-size: 1.2em;">Avg. Lead Time</h3>
                <h1 style="color: #e65100; margin:0; font-size: 2em; font-family: Canela;">{avg_lead_time_year:.1f}</h1>
                <p style="color: #ff9800; margin:0; font-size: 0.9em;">Days (Variable)</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # Plot: Continuous Line Chart
        
        # --- Date Range Logic ---
        st.markdown("<br>", unsafe_allow_html=True)
        view_opt = st.radio("Select History View Range:", ["2024 - 2025", "2021 - 2025"], index=0, horizontal=True)
        
        h_sub = hist_df[(hist_df['Part'] == sel_part) & (hist_df['Location'] == sel_loc)].sort_values('Date')
        
        if view_opt == "2024 - 2025":
            h_sub = h_sub[h_sub['Date'] >= '2024-01-01']
        
        if not h_sub.empty:
            last_hist_date = h_sub['Date'].iloc[-1]
            last_hist_val = h_sub['Value'].iloc[-1]
            f_dates = [last_hist_date] + list(f_sub['Date'])
            f_vals = [last_hist_val] + list(f_sub['Value'])
        else:
            f_dates = f_sub['Date'].tolist()
            f_vals = f_sub['Value'].tolist()

        # ECharts Implementation (Option 2) for Smooth Animation
        
        # Define missing variables
        h_dates = h_sub['Date'].tolist()
        h_vals = h_sub['Value'].tolist()
        
        # 1. Prepare Data
        # X-Axis: Unified Timeline
        # h_dates ends at T
        # f_dates starts at T
        
        # Convert to string for ECharts
        x_data = [d.strftime('%Y-%m-%d') for d in h_dates]
        # Append forecast dates (excluding the overlap start if it duplicates standard logic, 
        # but our f_dates logic includes overlap at index 0. So we slice [1:])
        if len(f_dates) > 1:
            x_data += [d.strftime('%Y-%m-%d') for d in f_dates[1:]]
        
        # Y-Axis Series
        # History: defined up to T, then null
        # Forecast: null up to T, then defined
        
        # Pad History
        # Length of forecast part to add is len(f_dates) - 1
        future_len = len(f_dates) - 1 if len(f_dates) > 0 else 0
        y_hist = h_vals + [None] * future_len
        
        # Pad Forecast
        # Needs to align at T (index len(h_dates)-1)
        # So we pad with None for len(h_dates)-1 positions
        padding_len = len(h_dates) - 1 if len(h_dates) > 0 else 0
        y_forecast = [None] * padding_len + f_vals
        
        # 2. Define Options
        options = {
            "title": {
                "text": "2025 Forecast Demand", 
                "left": "center", 
                "textStyle": {"fontFamily": "Canela", "fontSize": 24, "color": "#000"}
            },
            "legend": {
                "data": ["History", f"Forecast ({best_model_name})"],
                "bottom": "0",
                "textStyle": {"fontFamily": "Canela", "fontSize": 14}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross"}
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": x_data,
                "axisLine": {"lineStyle": {"color": "#ccc"}},
                "axisLabel": {"color": "#333"}
            },
            "yAxis": {
                "type": "value",
                "axisLine": {"show": False},
                "axisTick": {"show": False},
                "splitLine": {"show": True, "lineStyle": {"type": "dashed", "color": "#eee"}},
                "axisLabel": {"color": "#333"}
            },
            "series": [
                {
                    "name": "History",
                    "type": "line",
                    "data": y_hist,
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"width": 3, "color": "black"},
                    "itemStyle": {"color": "black"},
                    "animationDuration": 3000,
                    "animationEasing": "linear"
                },
                {
                    "name": f"Forecast ({best_model_name})",
                    "type": "line",
                    "data": y_forecast,
                    "smooth": True,
                    "showSymbol": True,
                    "symbol": "circle",
                    "symbolSize": 8,
                    "lineStyle": {"width": 4, "color": "#2ecc71"},
                    "itemStyle": {"color": "#2ecc71", "borderColor": "#fff", "borderWidth": 2},
                    "animationDuration": 2000,
                    "animationEasing": "linear",
                    "animationDelay": 3000 # Starts after History (3000ms)
                }
            ],
            "animation": True
        }
        
        # Force re-render with unique key to ensure animation plays every time
        st_echarts(options=options, height="500px", key=f"dash_chart_{uuid.uuid4()}")
        
        # Download Excel
        dl_data = pd.DataFrame({
            'Spare Part ID': [sel_part] * len(f_sub),
            'Location': [sel_loc] * len(f_sub),
            'Month': f_sub['Date'].dt.strftime('%B %Y'),
            'Forecasted Demand': f_sub['Value'].round(2),
            'Variable Lead Time (Days)': f_sub['LeadTime_Var'].round(2)
        })
        
        # Use Centered Button
        _, c_dl, _ = st.columns([1, 1, 1])
        with c_dl:
            excel_file = to_excel(dl_data)
            st.download_button(
                label="Download Forecast Data (Excel)",
                data=excel_file,
                file_name=f"Forecast_2025_{sel_part}_{sel_loc}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    st.divider()

    st.divider()

    # --- 4. GRAPH 2: TRAINING & TESTING (EDUCATIONAL) ---
    st.markdown(f"<h2 style='text-align: center; font-family: Canela;'>Training & Testing Data Analysis</h2>", unsafe_allow_html=True)
    
    with st.expander("View Model Performance Details", expanded=False):
        st.markdown(f"**Best Performing Model:** {best_model_name} on {best_split_name}")
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        
        # Helper for color
        def metric_tile(label, value, color="#fafafa", text_color="#333"):
            return f'''
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
                <small style="color: {text_color}; font-weight: bold;">{label}</small>
                <h2 style="color: {text_color}; margin: 5px 0;">{value}</h2>
            </div>
            '''
        
        if best_info is not None:
            with m1: st.markdown(metric_tile("MAPE", f"{best_mape:.2%}", "#e8f5e9", "#2e7d32"), unsafe_allow_html=True) # Green
            with m2: st.markdown(metric_tile("RMSE", f"{best_rmse:.2f}", "#fff3e0", "#ef6c00"), unsafe_allow_html=True) # Amber
            with m3: st.markdown(metric_tile("Bias", f"{best_bias:.2f}", "#fff3e0", "#ef6c00"), unsafe_allow_html=True) # Amber
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison Chart
        # Load validation data from Dashboard_Database for this Split
        val_df = df[(df['Part'] == sel_part) & 
                    (df['Location'] == sel_loc) & 
                    (df['Split'] == best_split_name)]
        
        if not val_df.empty:
            val_df['Date'] = pd.to_datetime(val_df['Date'])
            val_df = val_df.sort_values('Date')
            
            fig2 = go.Figure()
            
            # Plot Actuals
            actuals = val_df[val_df['Model'] == 'Actual']
            fig2.add_trace(go.Scatter(x=actuals['Date'], y=actuals['Value'], name='Actual (Test)', line=dict(color='black', width=2)))
            
            # Plot All Models in this Split to show comparison
            models = val_df['Model'].unique()
            for m in models:
                if m == 'Actual': continue
                
                # Color logic: Green for winner, Amber/Grey for others
                color = '#2ecc71' if m == best_model_name else '#bdc3c7'
                width = 3 if m == best_model_name else 1
                opacity = 1.0 if m == best_model_name else 0.5
                
                m_data = val_df[val_df['Model'] == m]
                fig2.add_trace(go.Scatter(x=m_data['Date'], y=m_data['Value'], name=m, 
                                          line=dict(color=color, width=width), opacity=opacity))
            
            fig2.update_layout(
                title=f"Model Comparison on {best_split_name}",
                xaxis_title="Date",
                yaxis_title="Demand",
                template="plotly_white",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    st.divider()
    
    # 5. Chart 2: Train/Test Split & Models (Tiles)
    # 5. Chart 2: Train/Test Split & Models (Tiles)
    with st.expander("Model Architecture & Training Splits", expanded=False):
        # Splits Info
        splits = [
            {"name": "Strategic Split", "ratio": "3y Train / 1y Test", "dates": "2021-2023 / 2024"},
            {"name": "Standard Split", "ratio": "3.2y Train / 0.8y Test", "dates": "Jan'21-Mar'24 / Apr'24-Dec'24"},
            {"name": "Tactical Split", "ratio": "3.5y Train / 0.5y Test", "dates": "Jan'21-Jun'24 / Jul'24-Dec'24"}
        ]
        
        # Models Info
        models = ["ETS (Holt-Winters)", "SARIMA", "Prophet", "XGBoost", "N-HiTS (Deep Learning)", "Weighted Ensemble"]
        
        # Layout: Tiles
        cols = st.columns(3)
        for i, sp in enumerate(splits):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; text-align: center;">
                    <h4 style="color: #023e8a;">{sp['name']}</h4>
                    <p style="font-weight: bold; font-size: 18px;">{sp['ratio']}</p>
                    <p style="color: #666; font-size: 12px;">{sp['dates']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Models List
        st.markdown("**Ensemble Composition (Council of Models):**")
        m_cols = st.columns(6)
        for i, m in enumerate(models):
            with m_cols[i]:
                st.markdown(f"<div style='text-align: center; padding: 5px; background: #e9ecef; border-radius: 5px; font-size: 12px;'>{m}</div>", unsafe_allow_html=True)
                
        # Visualization of Splits (Gantt-like)
        # Simple Plotly Timeline
        split_df = pd.DataFrame([
            dict(Task="Strategic", Start='2021-01-01', Finish='2024-01-01', Type='Train'),
            dict(Task="Strategic", Start='2024-01-01', Finish='2025-01-01', Type='Test'),
            dict(Task="Standard", Start='2021-01-01', Finish='2024-03-01', Type='Train'),
            dict(Task="Standard", Start='2024-03-01', Finish='2025-01-01', Type='Test'),
            dict(Task="Tactical", Start='2021-01-01', Finish='2024-06-01', Type='Train'),
            dict(Task="Tactical", Start='2024-06-01', Finish='2025-01-01', Type='Test')
        ])
        
        fig_split = px.timeline(split_df, x_start="Start", x_end="Finish", y="Task", color="Type", 
                                color_discrete_map={'Train': '#3498db', 'Test': '#e74c3c'},
                                title="Cross-Validation Windows")
        fig_split.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_split, use_container_width=True)

# --- 6. ABOUT PAGE ---
def about_page():
    render_navbar()
    st.markdown("## About the Project")
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### Automobile Spare Parts Forecasting Solution (GSCM 2025-26)
        
        **Project Overview:**
        This platform is a comprehensive Demand Forecasting Solution designed for the automobile spare parts industry. It analyzes historical demand data for **11 critical SKUs** across **2 distinct locations** (Mumbai/Bangalore) to generate accurate future predictions. The system leverages a Council of Models approach, training multiple statistical and deep learning models (ETS, SARIMA, Prophet, N-HiTS) on three strategic cross-validation splits to ensure robustness.
        
        **Key Offerings:**
        - **Interactive Dashboard:** Executive-level view of 2025 demand, lead times, and model performance metrics.
        - **Forecasting Tool:** On-demand scenario planning allowing users to upload new datasets for instant analysis.
        - **Composite Scoring:** Smart model selection using a weighted mix of MAPE (40%), RMSE (40%), and Bias (20%).
        """)
        
        st.markdown("""
<br> <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #2ecc71;"> <b>Tech Mahindra x IIM Udaipur</b><br> Project made under the guidance of <b>Mr. Adarsh Uppoor (Tech Mahindra)</b> & <b>Prof. Rahul Pandey (IIM Udaipur)</b>. </div> <br> Made with Curiosity by: <br> <b>Deevyendu N Shukla & Vishesh Bhargava</b><br> IIM Udaipur GSCM 2025-26
        """, unsafe_allow_html=True)
        
    with c2:
        if os.path.exists(IIMU_LOGO):
            st.image(IIMU_LOGO, width=200)
        st.info("Version 2.1 (Stable)")

# --- 5. FORECASTING TOOL PAGE ---
def forecasting_tool_page():
    render_navbar()
    st.markdown("""
    <style>
    /* Hover Glow for Download Button */
    .stDownloadButton > button:hover {
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.8), 0 0 30px rgba(46, 204, 113, 0.4) !important;
        border-color: #2ecc71 !important;
        color: #000000 !important;
        background-color: #ffffff !important;
        transition: all 0.3s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## AI/ML Forecasting Tool")
    st.divider()
    
    # File Upload
    uploaded_file = st.file_uploader("Upload Historical Data (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Validation
            is_valid, error_msg = fe.validate_columns(df)
            if not is_valid:
                st.error(error_msg)
                return
                
            # Clean Data
            df = fe.clean_data(df)
            st.success("File uploaded successfully! Select a Part/Location to Analyze.")
            
            # Selectors
            parts = df['Spare Part ID'].unique()
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                sel_part = st.selectbox("Select Part", parts)
            with c2:
                locs = df[df['Spare Part ID'] == sel_part]['Location'].unique()
                sel_loc = st.selectbox("Select Location", locs)
            
            # Analyze Button
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_btn = st.button("Analyze & Forecast", type="primary", use_container_width=True)
                
            if analyze_btn:
                progress_bar = st.progress(0, text="Starting Analysis...")
                
                # Run Engine
                def update_progress(p, text):
                    progress_bar.progress(p, text=text)
                    
                result = fe.analyze_part_location(df, sel_part, sel_loc, update_progress)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    progress_bar.progress(100, text="Analysis Complete!")
                    
                    # --- DISPLAY RESULTS ---
                    st.divider()
                    st.markdown(f"### Forecast Results: {sel_part} - {sel_loc}")
                    st.markdown(f"**Best Model Selected:** {result['best_model']} (RMSE: {result['best_rmse']:.2f})")
                    
                    # Tiles
                    t1, t2 = st.columns(2)
                    with t1:
                         st.markdown(f"""
                        <div style="background-color: #e8f5e9; padding: 10px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h3 style="color: #2e7d32; margin:0; font-family: Canela; font-size: 1.2em;">Forecasted Demand (12M)</h3>
                            <h1 style="color: #1b5e20; margin:0; font-size: 2em; font-family: Canela;">{result['total_forecast_demand']:,.0f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    with t2:
                        st.markdown(f"""
                        <div style="background-color: #fff3e0; padding: 10px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h3 style="color: #ef6c00; margin:0; font-family: Canela; font-size: 1.2em;">Avg. Lead Time (Forecast)</h3>
                            <h1 style="color: #e65100; margin:0; font-size: 2em; font-family: Canela;">{result['avg_forecast_lead_time']:.1f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ECharts Visualization
                    # Prepare Data
                    h_dates = result['history_dates']
                    f_dates = result['forecast_dates']
                    h_vals = result['history_values']
                    f_vals = result['forecast_values']
                    
                    # Convert dates
                    x_data = [d.strftime('%Y-%m-%d') for d in h_dates]
                    x_data += [d.strftime('%Y-%m-%d') for d in f_dates]
                    
                    # Series
                    y_hist = h_vals + [None] * len(f_vals)
                    y_forc = [None] * len(h_vals) + f_vals
                    
                    options = {
                        "title": {"text": "Forecast Visualization", "left": "center", "textStyle": {"fontFamily": "Canela"}},
                        "legend": {"data": ["History", "Forecast"], "bottom": "0", "textStyle": {"fontFamily": "Canela", "fontSize": 14}},
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "category", "data": x_data, "boundaryGap": False},
                        "yAxis": {"type": "value", "scale": True},
                        "grid": {"bottom": "10%", "containLabel": True},
                        "series": [
                            {
                                "name": "History", "type": "line", "data": y_hist, 
                                "smooth": True, "lineStyle": {"color": "black", "width": 3},
                                "itemStyle": {"color": "black"}, 
                                "animationDuration": 3000,
                                "animationEasing": "linear"
                            },
                            {
                                "name": "Forecast", "type": "line", "data": y_forc,
                                "smooth": True, "lineStyle": {"color": "#2ecc71", "width": 4},
                                "itemStyle": {"color": "#2ecc71"}, 
                                "animationDuration": 2000,
                                "animationEasing": "linear",
                                "animationDelay": 3000
                            }
                        ],
                        "animation": True
                    }
                    st_echarts(options=options, height="450px", key=f"tool_chart_{uuid.uuid4()}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# --- ROUTER ---
def main():
    # 1. Init Session State
    if 'page' not in st.session_state:
        st.session_state['page'] = 'landing'
        
    # 2. Sidebar Removed (Top Nav Implemented in pages)
    
    # 3. Page Rendering
    if st.session_state['page'] == 'landing':
        landing_page()
    elif st.session_state['page'] == 'dashboard':
        dashboard_page()
    elif st.session_state['page'] == 'tool':
        forecasting_tool_page()
    elif st.session_state['page'] == 'about':
        about_page()

if __name__ == "__main__":
    main()
