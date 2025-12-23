
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import os
import textwrap
import subprocess

# Config
# DB_FILE = 'Dashboard_Database.csv' # This constant is no longer needed as the path is hardcoded in load_data

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Automobile Spare Parts Forecasting")
STATUS_FILE = 'generation_status.json'

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Dashboard_Database.csv')
        
        # --- Pre-calculate Bias & Score ---
        # 1. Calculate Bias
        if 'Value' in df.columns:
            actuals = df[df['Model'] == 'Actual'][['Part', 'Location', 'Split', 'Date', 'Value']].rename(columns={'Value': 'Act'})
            forecasts = df[df['Model'] != 'Actual'][['Part', 'Location', 'Split', 'Model', 'Date', 'Value']].rename(columns={'Value': 'Fcst'})
            
            merged = pd.merge(forecasts, actuals, on=['Part', 'Location', 'Split', 'Date'], how='left')
            merged['Err'] = merged['Fcst'] - merged['Act']
            bias_df = merged.groupby(['Part', 'Location', 'Split', 'Model'])['Err'].mean().reset_index().rename(columns={'Err': 'Bias'})
            
            # Merge Bias back
            df = pd.merge(df, bias_df, on=['Part', 'Location', 'Split', 'Model'], how='left')
            df['Bias'] = df['Bias'].fillna(0) # For Actuals or missing
            
            # 2. Calculate Composite Score
            # Initialize Score
            df['Score'] = 999.0
            
            try:
                # Normalize per group (Part, Location, Split)
                # We need to act on a summary (1 row per model) then map back
                # Ensure Train_MAPE is in cols if it exists
                cols = ['Part', 'Location', 'Split', 'Model', 'MAPE', 'RMSE', 'Bias']
                if 'Train_MAPE' in df.columns:
                    cols.append('Train_MAPE')
                    
                summary = df.drop_duplicates(subset=['Part', 'Location', 'Split', 'Model'])[cols]
                summary = summary[summary['Model'] != 'Actual'] # Don't score actuals
                
                def calc_score(g):
                    try:
                        # Min-Max Normalization (Small is good for Score)
                        # MAPE (Test)
                        mn, mx = g['MAPE'].min(), g['MAPE'].max()
                        d = mx - mn
                        g['n_mape'] = (g['MAPE'] - mn) / d if d > 0 else 0
                        
                        # RMSE
                        mn, mx = g['RMSE'].min(), g['RMSE'].max()
                        d = mx - mn
                        g['n_rmse'] = (g['RMSE'] - mn) / d if d > 0 else 0
                        
                        # Abs(Bias)
                        g['abs_bias'] = g['Bias'].abs()
                        mn, mx = g['abs_bias'].min(), g['abs_bias'].max()
                        d = mx - mn
                        g['n_bias'] = (g['abs_bias'] - mn) / d if d > 0 else 0
                        
                        # Score formula: 0.7 MAPE + 0.2 RMSE + 0.1 Bias
                        # (User didn't ask to change formula, just to see Train MAPE)
                        g['Score'] = 0.7 * g['n_mape'] + 0.2 * g['n_rmse'] + 0.1 * g['n_bias']
                    except:
                        g['Score'] = 999.0
                    return g

                if not summary.empty:
                    # Fix: Normalize across ALL splits for the same Part/Location
                    # This ensures a 10% MAPE in Split A is better than 50% MAPE in Split B
                    scored = summary.groupby(['Part', 'Location']).apply(calc_score).reset_index(drop=True)
                    scored = scored[['Part', 'Location', 'Split', 'Model', 'Score']]
                    
                    # Remove default score col before merge to avoid _x _y collision
                    df = df.drop(columns=['Score'])
                    df = pd.merge(df, scored, on=['Part', 'Location', 'Split', 'Model'], how='left')
                    df['Score'] = df['Score'].fillna(999.0)
            except Exception as inner_e:
                # If scoring fails, we still return the DF, just with Score=999.0
                print(f"Scoring Calculation Failed: {inner_e}")
                pass

        return df
    except Exception as e:
        print(f"Data Load Error: {e}")
        return pd.DataFrame(columns=['Part', 'Location', 'Split', 'Model', 'Date', 'Value', 'MAPE', 'RMSE', 'Bias', 'Score'])

def get_progress():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def main():
    # st.set_page_config moved to module level
    
    # --- GLOBAL CSS (Canela Font) ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');
    
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Canela', 'Playfair Display', serif !important;
    }
    
    /* Sidebar specific */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Canela', 'Playfair Display', serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- BANNER REMOVED (Moved to bottom) ---
    # st.markdown(..., unsafe_allow_html=True)
    
    st.title("Automobile Spare Parts Forecasting Dashboard")
    
    # Removed st.sidebar.title("Forecasting Lab") if it existed here, 
    # but based on grep it was somewhere. Let's find where it was.
    # Ah, grep found it, but I didn't see it in lines 120-160.
    # It must be earlier or later.
    # I will search for it specifically to remove it.
    
    df = load_data()
    
    # --- SIDEBAR & PROGRESS ---
    # st.sidebar.title("Forecasting Lab") # Removed

    # Progress Indicator
    prog = get_progress()
    if prog and prog.get('percent', 0) < 100:
        st.sidebar.info(f"**Generating Data...**\n\n{prog.get('percent')}% Complete\n\n*{prog.get('message')}*")
        st.sidebar.progress(prog.get('percent')/100)
        if st.sidebar.button("Refresh Progress"):
            st.rerun()
    elif prog:
        st.sidebar.success("Generation Complete!")

    # st.sidebar.subheader("Configuration") # Removed Duplicate Header
    
    if df.empty:
        st.warning("⚠️ Data is generating... Please wait and refresh in a few minutes.")
        
        # Show progress bar in main area if data is missing/loading
        prog = get_progress()
        if prog:
            st.info(f"**Progress:** {prog.get('percent', 0)}% Complete\n\n*{prog.get('message', '')}*")
            st.progress(prog.get('percent', 0)/100)
        
        if st.button("Refresh Data"):
            st.rerun()
        return

    # Sidebar
    
    # 1. About Link (Top Left)
    if os.path.exists("pages/About.py"):
        try:
            st.sidebar.page_link("pages/About.py", label="About") # Removed icon
        except KeyError:
             st.sidebar.warning("⚠️ Restart app to enable 'About'")

    # 2. Main Title (Replaced Forecasting Lab)
    st.sidebar.markdown("<h1 style='font-size: 28px; font-weight: bold; margin-bottom: 20px;'>Modifications</h1>", unsafe_allow_html=True)
    
    # st.sidebar.header("Configuration") # Removed per request
    
    # About Link Moved to Top
    # if os.path.exists("pages/About.py")...
        
    # Local-Only Controls
    # Only show if specific user or environment indicates local dev
    is_local = os.path.exists("/Users/deevyendunshukla")
    
    if is_local:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Local Admin")
        
        # 1. Reload Data
        if st.sidebar.button("Reload Data Source"):
            load_data.clear()
            st.cache_data.clear()
            st.rerun()

        # 2. Deploy to Cloud
        if st.sidebar.button("Deploy to Cloud 🚀"):
            with st.sidebar.status("Deploying to GitHub...", expanded=True) as status:
                try:
                    # Add
                    status.write("Staging files...")
                    subprocess.run(["git", "add", "."], check=True)
                    
                    # Commit
                    status.write("Committing...")
                    # Allow empty commit to fail gracefully (or check status first)
                    result = subprocess.run(["git", "commit", "-m", "Update from Dashboard Button"], capture_output=True, text=True)
                    if result.returncode != 0 and "nothing to commit" in result.stdout:
                         status.write("Nothing to commit (already up to date).")
                    elif result.returncode != 0:
                        status.update(label="Commit Failed", state="error")
                        st.sidebar.error(f"Commit Error: {result.stderr}")
                    
                    # Push
                    status.write("Pushing to Cloud...")
                    subprocess.run(["git", "push"], check=True)
                    
                    status.update(label="Deployment Complete!", state="complete")
                    st.sidebar.success("Changes pushed! Cloud update in ~2 mins.")
                    
                except subprocess.CalledProcessError as e:
                    status.update(label="Deployment Failed", state="error")
                    st.sidebar.error(f"Git Error: {e}")
                except Exception as e:
                    st.sidebar.error(f"Unexpected Error: {e}")
    
    # ETS Control
    
    # ETS Control
    enable_ets = st.sidebar.checkbox("Enable ETS (Holt-Winters)", value=False)
    if not enable_ets:
        df = df[df['Model'] != 'ETS']
    
    # 1. Part & Location
    parts = df['Part'].unique()
    sel_part = st.sidebar.selectbox("Spare Part", parts)
    
    locs = df[df['Part'] == sel_part]['Location'].unique()
    locs = df[df['Part'] == sel_part]['Location'].unique()
    sel_loc = st.sidebar.selectbox("Location", locs)
    
    # State for auto-selection
    if 'last_part' not in st.session_state:
        st.session_state['last_part'] = sel_part
    if 'last_loc' not in st.session_state:
        st.session_state['last_loc'] = sel_loc

    # --- OPTIMIZATION INSIGHT ---
    # Find the best split/model combination globally based on Weighted Score
    best_fit_split = None
    best_fit_model = None
    best_fit_score = 999.0
    best_fit_mape = 0.0 # Just for display
    best_fit_train_mape = 0.0 # For display
    
    # Filter for this part/loc regardless of split
    # Ensure Score exists
    if 'Score' not in df.columns:
        df['Score'] = 999.0
        
    global_subset = df[(df['Part'] == sel_part) & (df['Location'] == sel_loc) & (df['Model'] != 'Actual')]
    
    for _, row in global_subset.iterrows():
        # Iterate unique combinations
        if row['Score'] < best_fit_score:
            best_fit_score = row['Score']
            best_fit_model = row['Model']
            best_fit_split = row['Split']
            best_fit_split = row['Split']
            best_fit_mape = row['MAPE']
            best_fit_train_mape = row['Train_MAPE'] if 'Train_MAPE' in row else 0.0

    # Auto-switch to Best Fit if Part/Loc changed
    if sel_part != st.session_state['last_part'] or sel_loc != st.session_state['last_loc']:
        if best_fit_split:
            st.session_state['sel_split_state'] = best_fit_split
            st.toast(f"🔄 Auto-switched to Best Fit: {best_fit_split}", icon="✨")
        st.session_state['last_part'] = sel_part
        st.session_state['last_loc'] = sel_loc
        # Rerun to ensure the Radio button picks up the new state immediately if needed, 
        # though setting state before widget might be enough if we use key/index correctly.
        # But st.radio index comes from logic below. Let's just let it flow.
            
    if best_fit_split:
        st.sidebar.markdown("---")
    if best_fit_split:
        st.sidebar.markdown("---")
        st.sidebar.info(f"✨ **Recommendation**\n\nOptimal Strategy: **{best_fit_split}**\nModel: **{best_fit_model}**\nTest MAPE: **{best_fit_mape:.2%}**\nTrain MAPE: **{best_fit_train_mape:.2%}**\n*(Based on Composite Score)*")
        if st.sidebar.button("Apply Best Fit"):
            st.session_state['sel_split_state'] = best_fit_split
            st.rerun()

    # 2. Split Strategy
    splits = df['Split'].unique()
    # Handle state override
    default_split_idx = 0
    if best_fit_split and 'sel_split_state' not in st.session_state:
         # Initial load default
         if best_fit_split in splits:
             default_split_idx = list(splits).index(best_fit_split)
    elif 'sel_split_state' in st.session_state and st.session_state['sel_split_state'] in splits:
         default_split_idx = list(splits).index(st.session_state['sel_split_state'])
         
    # Key is important to sync with session state, but we are manually managing index.
    # Actually, best way: output the widget, then if user changes it, update state?
    # Or just use key='sel_split_state' if we trust it?
    # Mixing manual index and key is tricky. Let's stick to index control.
    def on_split_change():
        st.session_state['sel_split_state'] = st.session_state.split_radio
        
    sel_split = st.sidebar.radio("Training Strategy", splits, index=default_split_idx, key='split_radio', on_change=on_split_change)
    
    # Filter Data
    subset = df[
        (df['Part'] == sel_part) & 
        (df['Location'] == sel_loc) & 
        (df['Split'] == sel_split)
    ]
    
    # 3. Model Selection
    avail_models = [m for m in subset['Model'].unique() if m != 'Actual']
    
    # Safe Defaults
    wanted_defaults = ['SARIMA', 'Weighted Ensemble', 'Prophet', 'XGBoost', 'N-HiTS', 'ETS']
    valid_defaults = [m for m in wanted_defaults if m in avail_models]
    if not valid_defaults and avail_models:
        valid_defaults = [avail_models[0]]
        
    sel_models = st.sidebar.multiselect("Select Models to Compare", avail_models, default=valid_defaults)
    
    # --- METRICS SECTION ---
    st.subheader(f"Performance Metrics ({sel_split})")
    
    # Calculate Best Overall based on Score
    best_overall_score = 999.0
    best_overall_model = ""
    best_overall_mape = 0.0
    
    for m in avail_models:
        m_subset = subset[subset['Model'] == m]
        if not m_subset.empty:
            curr_score = m_subset.iloc[0]['Score'] if 'Score' in m_subset.columns else 999.0
            if curr_score < best_overall_score:
                best_overall_score = curr_score
                best_overall_model = m
                best_overall_mape = m_subset.iloc[0]['MAPE']
    
    if best_overall_model:
        # Check if this local winner is also the global best fit
        is_global_best = (best_overall_model == best_fit_model) and (sel_split == best_fit_split)
        
        if is_global_best:
            st.success(f"**Global Winner:** **{best_overall_model}** yielded the best results overall using the **{sel_split}** training/testing period. (MAPE: {best_overall_mape:.2%})")
        else:
            global_hint = f" (Global Winner is {best_fit_model} in {best_fit_split})" if best_fit_split else ""
            st.warning(f"**Split Winner:** **{best_overall_model}** is the best in this split ({sel_split}).{global_hint}")

    # --- CSS STYLES ---
    # 1. Dynamic Background based on Global Best Status
    bg_color = "linear-gradient(to bottom, #d4edda, #ffffff)" if is_global_best else "linear-gradient(to bottom, #fff3cd, #ffffff)"
    
    st.markdown(f"""
    <style>
    .stApp {{
        background: {bg_color};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # 2. Static Styles (Metric Cards, etc.)
    st.markdown("""
    <style>
    /* Ensure sidebar stays clean */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        width: 220px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #ddd;
    }
    .winner-card {
        background-color: #d4edda !important; 
        border: 2px solid #28a745 !important;
        color: #155724 !important;
    }
    .metric-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-sub {
        font-size: 14px;
        color: #555;
    }
    .winner-card .metric-sub {
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- HTML METRICS ---
    html_cards = '<div class="metric-container">'
    
    for i, model in enumerate(sel_models):
        m_data = subset[subset['Model'] == model]
        if m_data.empty: continue
        
        # Metrics are repeated in rows, just take first
        mape = m_data.iloc[0]['MAPE']
        rmse = m_data.iloc[0]['RMSE']
        # Use pre-calculated Bias and Score
        bias = m_data.iloc[0]['Bias'] if 'Bias' in m_data.columns else 0.0
        score = m_data.iloc[0]['Score'] if 'Score' in m_data.columns else 0.0
        train_mape = m_data.iloc[0]['Train_MAPE'] if 'Train_MAPE' in m_data.columns else 0.0

        if model == 'Weighted Ensemble' and mape >= 0.99:
            continue

        is_winner = (model == best_overall_model)
        card_class = "metric-card winner-card" if is_winner else "metric-card"
        trophy = ""
        
        html_cards += f"""
<div class="{card_class}">
    <div class="metric-title">{trophy}{model}</div>
    <div class="metric-value" style="font-size: 20px;">Test: {mape:.2%}</div>
    <div class="metric-sub" style="font-weight:bold; margin-bottom:5px;">Train: {train_mape:.2%}</div>
    <div class="metric-sub">RMSE: {rmse:.1f}</div>
    <div class="metric-sub">Bias: {bias:.1f}</div>
    <div class="metric-sub" style="font-size:12px; margin-top:5px">Score: {score:.3f}</div>
</div>
"""
    
    html_cards += '</div>'
    st.markdown(html_cards, unsafe_allow_html=True)

    # --- CHART SECTION ---
    st.subheader("Forecast for Testing Period")
    
    fig = go.Figure()
    
    # 1. Actuals
    actuals = subset[subset['Model'] == 'Actual'].sort_values('Date')
    fig.add_trace(go.Scatter(
        x=actuals['Date'], y=actuals['Value'],
        mode='lines+markers', name='Actual Demand',
        line=dict(color='black', width=3)
    ))
    
    # 2. Models
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F6', '#FFC300', '#00BCD4', '#9C27B0']
    for i, model in enumerate(sel_models):
        m_data = subset[subset['Model'] == model].sort_values('Date')
        if m_data.empty: continue
        
        # Get metrics for label
        mape_val = m_data.iloc[0]['MAPE']
        train_mape_val = m_data.iloc[0]['Train_MAPE'] if 'Train_MAPE' in m_data.columns else 0.0
        
        is_winner = (model == best_overall_model)
        
        # Customize Trace Name with Metrics
        if train_mape_val > 0:
            label = f"{model} (Train: {train_mape_val:.1%} | Test: {mape_val:.1%})"
        else:
            label = f"{model} (Test MAPE: {mape_val:.1%})"
        
        if is_winner:
            # Highlight Winner
            opacity = 1.0
            width = 4
            color = '#28a745' # Success Green
            dash = 'solid'
            # label = f"🏆 {label}" # Removed emoji per request
        else:
            # Dim others
            opacity = 0.3
            width = 1.5
            color = '#cccccc' # Light Grey
            base_color = colors[i % len(colors)]
            color = base_color
            dash = 'dot'
        
        fig.add_trace(go.Scatter(
            x=m_data['Date'], y=m_data['Value'],
            mode='lines', name=label,
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity
        ))
        
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("*Note: 'Train' reflects in-sample fitted error (Overfitting Check). 'Test' reflects out-of-sample forecast error.*")
    
    # --- RAW DATA ---
    with st.expander("View Raw Data"):
        st.dataframe(subset)

    # --- 2025 OUTLOOK SECTION ---
    st.markdown("---")
    st.header("2025 Demand Projection")
    
    FUTURE_DB = 'Future_Forecast_Database.csv'
    if os.path.exists(FUTURE_DB):
        try:
            future_df = pd.read_csv(FUTURE_DB)
            # Filter for Part, Location AND the determined Global Winner Model
            # This ensures consistency with the banner.
            winner_model_name = best_overall_model
            f_sub = future_df[
                (future_df['Part'] == sel_part) & 
                (future_df['Location'] == sel_loc) & 
                (future_df['Model'] == winner_model_name)
            ].sort_values('Date')
            
            if not f_sub.empty:
                # Prepare History (Actuals)
                # We need full history for context
                # We need full history for context
                # Get from 'df' (the loaded dashboard db) -> filter Actuals
                # But 'df' only has Split data. We might have gaps if splits don't cover everything or overlap.
                # Ideally we want the "Latest" Split's Actuals?
                # Let's take 'Actual' rows from the 'subset' (current view) but that depends on selected split.
                # To show nice history, we should probably take Actuals from ALL available splits in df, deduplicated.
                
                # Fetch all actuals for this Part/Loc from generated DB
                all_actuals = df[(df['Part'] == sel_part) & (df['Location'] == sel_loc) & (df['Model'] == 'Actual')]
                all_actuals = all_actuals.drop_duplicates(subset=['Date']).sort_values('Date')
                
                # Combine
                f_sub['Date'] = pd.to_datetime(f_sub['Date'])
                all_actuals['Date'] = pd.to_datetime(all_actuals['Date'])
                
                # Stats
                total_2025 = f_sub['Value'].sum()
                
                # 2024 Total (from Actuals)
                mask_2024 = (all_actuals['Date'] >= '2024-01-01') & (all_actuals['Date'] <= '2024-12-31')
                total_2024 = all_actuals[mask_2024]['Value'].sum()
                
                growth = 0.0
                if total_2024 > 0:
                    growth = (total_2025 - total_2024) / total_2024
                
                # Metric Tiles
                c1, c2, c3 = st.columns(3)
                c1.metric("Selected Best Global Model", winner_model_name)
                c2.metric("Projected Demand (2025)", f"{int(total_2025):,}")
                c3.metric("Growth vs 2024", f"{growth:+.1%}")
                
                # Chart
                fig2 = go.Figure()
                
                # 1. History (Actuals)
                fig2.add_trace(go.Scatter(
                    x=all_actuals['Date'], y=all_actuals['Value'],
                    mode='lines', name='Historical Demand (Actual)',
                    line=dict(color='black', width=2)
                ))
                
                # 2. Recent Test Forecast (Context)
                # Find the test predictions for this model from the DB to show "recent performance"
                # We prioritize the latest split to show the immediate past
                # Filter df for this model
                model_past = df[(df['Part'] == sel_part) & (df['Location'] == sel_loc) & (df['Model'] == winner_model_name)]
                
                # Pick the split that ends latest (max date)
                if not model_past.empty:
                    # Find split with max date
                    latest_split = model_past.loc[model_past['Date'].idxmax()]['Split']
                    recent_preds = model_past[model_past['Split'] == latest_split].sort_values('Date')
                    
                    fig2.add_trace(go.Scatter(
                        x=recent_preds['Date'], y=recent_preds['Value'],
                        mode='lines', name=f'Recent Test Forecast ({winner_model_name})',
                        line=dict(color='#28a745', width=2, dash='dot'),
                        opacity=0.7
                    ))
                    
                    # Connect lines: Add last point of recent_preds to start of f_sub
                    if not recent_preds.empty and not f_sub.empty:
                        last_pt = recent_preds.iloc[-1]
                        # Create a row. Ensure columns match or just construct df
                        # We just need Date and Value for the chart
                        connector = pd.DataFrame({
                            'Date': [last_pt['Date']], 
                            'Value': [last_pt['Value']]
                        })
                        f_sub = pd.concat([connector, f_sub], axis=0)

                # 3. 2025 Forecast
                fig2.add_trace(go.Scatter(
                    x=f_sub['Date'], y=f_sub['Value'],
                    mode='lines+markers', name=f'2025 Forecast ({winner_model_name})',
                    line=dict(color='#28a745', width=3, dash='solid')
                ))
                
                fig2.update_layout(height=400, xaxis_title="Date", yaxis_title="Demand", title="History & 2025 Forecast")
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                 st.info("No 2025 forecast generated for this Part/Location yet.")
        except Exception as e:
            st.error(f"Error loading forecast: {e}")
    else:
        st.warning("Future Forecast Database not found. Please regenerate data.")

    # --- ABOUT SECTION REMOVED (Moved to pages/About.py) ---

if __name__ == "__main__":
    main()
