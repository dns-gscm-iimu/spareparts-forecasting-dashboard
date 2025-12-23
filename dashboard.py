
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configuration
INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
METRICS_FILE = 'Model_Performance.csv'
FORECAST_FILE = 'Forecast_Results.xlsx'

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Load Historical
    df_hist = pd.read_excel(INPUT_FILE, usecols=['Part ID', 'Location', 'Month', 'Demand'])
    df_hist['Month'] = pd.to_datetime(df_hist['Month'])
    
    # Load Metrics
    try:
        df_metrics = pd.read_csv(METRICS_FILE)
    except:
        df_metrics = pd.DataFrame()
        
    # Load Forecasts
    try:
        df_forecast = pd.read_excel(FORECAST_FILE)
    except:
        df_forecast = pd.DataFrame()
        
    return df_hist, df_metrics, df_forecast

def main():
    st.title("🛡️ Spare Parts Demand Forecasting")
    
    df_hist, df_metrics, df_forecast = load_data()
    
    if df_hist.empty:
        st.error("Historical data not found. Please run the forecasting engine first.")
        return

    # Sidebar Selection
    st.sidebar.header("Selection")
    
    parts = df_hist['Part ID'].unique()
    selected_part = st.sidebar.selectbox("Select Part ID", parts)
    
    locations = df_hist[df_hist['Part ID'] == selected_part]['Location'].unique()
    selected_loc = st.sidebar.selectbox("Select Location", locations)
    
    # Filter Data
    hist_data = df_hist[(df_hist['Part ID'] == selected_part) & (df_hist['Location'] == selected_loc)]
    hist_data = hist_data.set_index('Month').sort_index()
    
    # METRICS SECTION
    st.header(f"Analysis for {selected_part} - {selected_loc}")
    
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Historical Demand & Best Forecast")
        
        # Plot History
        fig = go.Figure()
        
        # Actual History
        fig.add_trace(go.Scatter(
            x=hist_data.index, 
            y=hist_data['Demand'], 
            mode='lines+markers',
            name='Historical Demand',
            line=dict(color='blue')
        ))
        
        # Forecast (if available)
        if not df_forecast.empty:
            forecast_row = df_forecast[
                (df_forecast['Part ID'] == selected_part) & 
                (df_forecast['Location'] == selected_loc)
            ]
            
            if not forecast_row.empty:
                # Extract monthly columns (assuming format 'Jan-2025' etc)
                # We identify forecast columns by checking they are NOT the info cols
                info_cols = ['Part ID', 'Location', 'Best Model', 'RMSE (Test)', 'MAPE (Test)']
                date_cols = [c for c in forecast_row.columns if c not in info_cols]
                
                forecast_vals = forecast_row.iloc[0][date_cols]
                forecast_dates = pd.to_datetime(date_cols, format='%b-%Y')
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_vals,
                    mode='lines+markers',
                    name=f"Forecast ({forecast_row.iloc[0]['Best Model']})",
                    line=dict(color='green', dash='dash')
                ))
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Comparison")
        if not df_metrics.empty:
            metrics_subset = df_metrics[
                (df_metrics['Part ID'] == selected_part) & 
                (df_metrics['Location'] == selected_loc)
            ].sort_values('RMSE')
            
            # Highlight best model
            st.dataframe(metrics_subset[['Model', 'RMSE', 'MAPE']].style.format({
                'RMSE': '{:.2f}',
                'MAPE': '{:.2%}'
            }).background_gradient(subset=['RMSE'], cmap='Greens_r'))
            
            best_model = metrics_subset.iloc[0]['Model'] if not metrics_subset.empty else "N/A"
            st.success(f"Best Performing Model: **{best_model}**")
        else:
            st.info("No metrics available yet.")

    # Data Table Expander
    with st.expander("View Raw Data"):
        st.write(hist_data)

if __name__ == "__main__":
    main()
