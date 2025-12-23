import pandas as pd

INPUT_FILE = 'Dashboard_Database.csv'
OUTPUT_FILE = 'Model_Comparison_Summary.xlsx'

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Dashboard_Database.csv not found.")
        return

    # Filter out Actuals
    df_models = df[df['Model'] != 'Actual'].copy()
    
    # Filter out invalid Weighted Ensembles (where MAPE is 1.0 due to 0 predictions)
    df_models = df_models[~((df_models['Model'] == 'Weighted Ensemble') & (df_models['MAPE'] >= 0.999))]

    # The metric is repeated for every date in the forecast period.
    # We just need one row per (Part, Location, Split, Model).
    summary = df_models.drop_duplicates(subset=['Part', 'Location', 'Split', 'Model'])[['Part', 'Location', 'Split', 'Model', 'MAPE', 'RMSE']]

    # Sort for readability
    summary = summary.sort_values(by=['Part', 'Location', 'Split', 'MAPE'])

    # --- Calculate Bias (Mean Forecast Error) ---
    # We need Actuals to compare.
    print("Calculating Bias...")
    df_actuals = df[df['Model'] == 'Actual'][['Part', 'Location', 'Split', 'Date', 'Value']].rename(columns={'Value': 'Actual'})
    df_forecasts = df[df['Model'] != 'Actual'][['Part', 'Location', 'Split', 'Model', 'Date', 'Value']].rename(columns={'Value': 'Forecast'})
    
    # Merge on Part, Location, Split, Date
    merged = pd.merge(df_forecasts, df_actuals, on=['Part', 'Location', 'Split', 'Date'], how='left')
    merged['Error'] = merged['Forecast'] - merged['Actual']
    
    # Group by Part, Location, Split, Model and calc Mean Error
    bias_df = merged.groupby(['Part', 'Location', 'Split', 'Model'])['Error'].mean().reset_index().rename(columns={'Error': 'Bias'})
    
    # Merge Bias back into summary
    summary = pd.merge(summary, bias_df, on=['Part', 'Location', 'Split', 'Model'], how='left')

    # --- Calculate Composite Score ---
    print("Calculating Composite Score...")
    # Normalize per group
    def calc_score_excel(g):
        # MAPE
        mn, mx = g['MAPE'].min(), g['MAPE'].max()
        d = mx - mn
        g['n_mape'] = (g['MAPE'] - mn) / d if d > 0 else 0
        
        # RMSE
        mn, mx = g['RMSE'].min(), g['RMSE'].max()
        d = mx - mn
        g['n_rmse'] = (g['RMSE'] - mn) / d if d > 0 else 0
        
        # Bias
        g['abs_bias'] = g['Bias'].abs()
        mn, mx = g['abs_bias'].min(), g['abs_bias'].max()
        d = mx - mn
        g['n_bias'] = (g['abs_bias'] - mn) / d if d > 0 else 0
        
        g['Score'] = 0.7 * g['n_mape'] + 0.2 * g['n_rmse'] + 0.1 * g['n_bias']
        return g

    summary = summary.groupby(['Part', 'Location']).apply(calc_score_excel).reset_index(drop=True)
    # Cleanup temp cols
    summary = summary.drop(columns=['n_mape', 'n_rmse', 'n_bias', 'abs_bias'])

    print("Creating Pivot Tables...")
    pivot_mape = summary.pivot_table(index=['Part', 'Location', 'Split'], columns='Model', values='MAPE')
    pivot_rmse = summary.pivot_table(index=['Part', 'Location', 'Split'], columns='Model', values='RMSE')
    pivot_bias = summary.pivot_table(index=['Part', 'Location', 'Split'], columns='Model', values='Bias')
    pivot_score = summary.pivot_table(index=['Part', 'Location', 'Split'], columns='Model', values='Score')

    print(f"Writing to {OUTPUT_FILE} (with highlights)...")
    
    # 1. Style Matrix: Highlight min MAPE/Score
    style_mape = pivot_mape.style.highlight_min(axis=1, color='yellow')
    style_score = pivot_score.style.highlight_min(axis=1, color='yellow')
    
    # 2. Style Flat Summary: Highlight the row with min SCORE
    def highlight_best_score(df):
        metrics = df.groupby(['Part', 'Location', 'Split'])['Score'].transform('min')
        is_best = df['Score'] == metrics
        return ['background-color: yellow' if v else '' for v in is_best]
    
    # Re-apply row highlighting logic properly
    summary['is_winner'] = summary.groupby(['Part', 'Location', 'Split'])['Score'].transform('min') == summary['Score']
    
    def highlight_winner_row(row):
        color = 'yellow' if row.get('is_winner') else ''
        return [f'background-color: {color}' for _ in row]

    style_flat = summary.style.apply(highlight_winner_row, axis=1)

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        style_flat.to_excel(writer, sheet_name='Flat Summary', index=False)
        style_mape.to_excel(writer, sheet_name='MAPE Matrix')
        pivot_rmse.to_excel(writer, sheet_name='RMSE Matrix')
        pivot_bias.to_excel(writer, sheet_name='Bias Matrix')
        style_score.to_excel(writer, sheet_name='Score Matrix')

    print("Done!")

if __name__ == "__main__":
    main()
