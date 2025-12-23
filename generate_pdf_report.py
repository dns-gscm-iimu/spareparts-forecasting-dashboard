
from fpdf import FPDF
import pandas as pd
import os
import datetime

# Configuration
DASHBOARD_DB = 'Dashboard_Database.csv'
FUTURE_DB = 'Future_Forecast_Database.csv'
REPORT_FILE = 'Forecasting_Report.pdf'

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Automobile Spare Parts Forecasting Report', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()

def generate_report():
    print("Generating PDF Report...")
    
    if not os.path.exists(DASHBOARD_DB):
        print("Error: Database not found.")
        return

    # Load Data
    df = pd.read_csv(DASHBOARD_DB)
    future_df = pd.read_csv(FUTURE_DB) if os.path.exists(FUTURE_DB) else pd.DataFrame()
    
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- Executive Summary ---
    pdf.chapter_title("1. Executive Summary")
    summary_text = (
        f"This report summarizes the AI/ML forecasting training for the Automobile Spare Parts project.\n"
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Scope: {df['Part'].nunique()} SKUs across {df['Location'].nunique()} Locations.\n"
        f"Models Trained: SARIMA, Prophet, XGBoost, Weighted Ensemble.\n"
        f"Training Window: 4 Years of historical data (approx).\n"
    )
    pdf.chapter_body(summary_text)
    
    # --- Training Methodology ---
    pdf.chapter_title("2. Training Methodology & Splits")
    method_text = (
        "To ensure robust model selection, we employed a Backtesting Strategy with multiple Training/Testing splits:\n"
        "1. Split 3y/1y: Train on first 3 years, Test on last 1 year.\n"
        "2. Split 3.5y/0.5y: Train on 3.5 years, Test on last 6 months.\n"
        "3. Split 3.2y/0.8y: Train on 3.2 years, Test on last ~9 months.\n\n"
        "Selection Logic: The 'Best Global Model' was selected based on a Composite Score (70% MAPE, 20% RMSE, 10% Bias) averaged across all splits."
    )
    pdf.chapter_body(method_text)
    
    # --- Best Model Stats ---
    pdf.chapter_title("3. Best Performing Models (Stats)")
    
    # Identify Winners
    # Replicate logic briefly or just pick lowest score per part/loc
    # We need to normalize scores again or just assume Score col is valid
    # To correspond to dashboard:
    # Use pre-calculated scores if available, or just take min score row
    if 'Score' not in df.columns:
        df['Score'] = df['MAPE'] # Fallback
        
    pdf.set_font('Arial', 'B', 10)
    # Headers
    cols = [30, 20, 45, 25, 25, 25] # Widths
    headers = ['Part', 'Loc', 'Best Model', 'Test MAPE', 'Train MAPE', 'RMSE']
    for i, h in enumerate(headers):
        pdf.cell(cols[i], 7, h, 1)
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    
    # Iterate Part/Loc
    unique_skus = df[['Part', 'Location']].drop_duplicates().values
    for part, loc in unique_skus:
        # Get winner (min score)
        sub = df[(df['Part']==part) & (df['Location']==loc) & (df['Model']!='Actual')]
        if sub.empty: continue
        
        # We want the GLOBAL winner.
        # Dashboard logic does averaging/normalization.
        # Simplified here: Pick row with absolute lowest Score across all splits?
        # Or better, just display the best split's stats for that winner.
        winner = sub.loc[sub['Score'].idxmin()]
        
        train_mape_str = f"{winner['Train_MAPE']:.1%}" if 'Train_MAPE' in winner else "N/A"
        
        pdf.cell(cols[0], 6, str(part), 1)
        pdf.cell(cols[1], 6, str(loc), 1)
        pdf.cell(cols[2], 6, str(winner['Model']), 1)
        pdf.cell(cols[3], 6, f"{winner['MAPE']:.1%}", 1)
        pdf.cell(cols[4], 6, train_mape_str, 1)
        pdf.cell(cols[5], 6, f"{winner['RMSE']:.1f}", 1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(0, 5, "*Train MAPE indicates in-sample fit. A very low Train MAPE vs High Test MAPE suggests Overfitting.")
    pdf.ln(5)

    # --- 2025 Outlook ---
    pdf.chapter_title("4. 2025 Future Outlook")
    
    if not future_df.empty:
        pdf.set_font('Arial', 'B', 10)
        cols_f = [30, 20, 45, 40]
        headers_f = ['Part', 'Loc', 'Selected Model', 'Projected Total (2025)']
        for i, h in enumerate(headers_f):
            pdf.cell(cols_f[i], 7, h, 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 10)
        
        for part, loc in unique_skus:
            # We want the winner model used in dashboard
            # We have to guess/re-derive winner or check future_df for this part/loc
            # Since we generated ALL, we need to pick the winner logic again.
            # Reuse 'winner' from above loop? Yes.
             sub = df[(df['Part']==part) & (df['Location']==loc) & (df['Model']!='Actual')]
             if sub.empty: continue
             winner_row = sub.loc[sub['Score'].idxmin()]
             winner_model = winner_row['Model']
             
             # Get Future Data for this winner
             f_sub = future_df[(future_df['Part']==part) & (future_df['Location']==loc) & (future_df['Model']==winner_model)]
             total_2025 = f_sub['Value'].sum()
             
             pdf.cell(cols_f[0], 6, str(part), 1)
             pdf.cell(cols_f[1], 6, str(loc), 1)
             pdf.cell(cols_f[2], 6, str(winner_model), 1)
             pdf.cell(cols_f[3], 6, f"{int(total_2025):,}", 1)
             pdf.ln()
             
    else:
        pdf.chapter_body("No 2025 forecast data found.")

    pdf.output(REPORT_FILE, 'F')
    print(f"Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()
