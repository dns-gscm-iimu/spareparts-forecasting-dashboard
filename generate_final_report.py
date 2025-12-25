from fpdf import FPDF
import os
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Demand Forecasting Project Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Times', '', 12)
        # Handle unicode replacement for standard fonts
        body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

    def add_code_chapter(self, title, file_path):
        self.add_page()
        self.chapter_title(title)
        self.set_font('Courier', '', 8)
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            # Basic sanitization
            code = code.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, code)
        except Exception as e:
            self.multi_cell(0, 5, f"Could not read file: {e}")

pdf = PDF()
pdf.set_title("Automobile Spare Parts Demand Forecasting")
pdf.set_author("Data Science Team")

# --- 1. TITLE PAGE ---
pdf.add_page()
pdf.set_font('Arial', 'B', 24)
pdf.ln(60)
pdf.cell(0, 10, "Demand Forecasting &", 0, 1, 'C')
pdf.cell(0, 10, "Inventory Optimization", 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Arial', '', 16)
pdf.cell(0, 10, "Detailed Methodology & Architecture Report", 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', 'I', 12)
pdf.cell(0, 10, f"Generated: {datetime.date.today()}", 0, 1, 'C')
pdf.ln(40)
# Summary
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, 
    "A comprehensive analysis of the AI/ML forecasting framework developed to predict spare parts demand across multiple distribution centers. This report details the data architecture, model selection constraints, feature engineering strategies, and the robust validation logic used to minimize supply chain risk."
, 0, 'C')

# --- 2. SCOPE ---
scope_text = """
1. Project Scope: Spare Parts & Locations

The fundamental objective of this initiative was to engineer a resilient demand forecasting system capable of navigating the stochastic nature of automobile spare parts consumption. Unlike Fast Moving Consumer Goods (FMCG), spare parts demand is often intermittent, lumpy, and driven by external factors such as vehicle breakdowns and maintenance schedules.

To validate our approach, we focused on a diverse subset of the inventory, specifically selecting five critical Stock Keeping Units (SKUs): PD457, PD2976, PD1399, PD3978, and PD238. These parts were not chosen at random; they represent a cross-section of the inventory classification matrix, incorporating high-value items (Class A), fast-moving consumables (Class F), and critical operational components.

Each SKU is managed across two distinct logistical nodes: Location A (Primary Distribution Hub) and Location B (Regional Warehouse). Location A typically exhibits higher volatility and volume, serving as a central aggregation point, whereas Location B shows more damped, delayed demand patterns typical of regional centers. In total, the forecasting engine manages 10 distinct time series (5 Parts × 2 Locations), each requiring a tailored modeling approach.
"""
pdf.add_chapter("1. Project Scope & Data Landscape", scope_text)

# --- 3. TRAINING & TESTING PERIOD ---
period_text = """
2. Training and Testing Strategy

A critical failure point in many forecasting projects is "overfitting"—where a model memorizes the historical noise rather than learning the underlying trend. To mitigate this, we strictly enforced a temporal separation between the data used for training and the data used for validation.

2.1 The 80:20 Evaluation Protocol
We partitioned the historical dataset (spanning January 2021 to December 2024) using the Pareto Principle.
The first 80% of the timeline (approx. Jan 2021 - Early 2024) was designated as the "Training Set." This substantial history is essential for the models to learn complex annual seasonalities (e.g., pre-monsoon maintenance spikes) and long-term trends.
The remaining 20% (approx. last 8-10 months of 2024) was sequestered as the "Testing Set." Crucially, the models were blind to this data during training. By evaluating performance on this unseen "future," we simulate how the model will perform in the real world.

2.2 Robustness via Heuristic Splits
Recognizing that a single split might be biased by a specific event (e.g., a stockout in mid-2024), we implemented two auxiliary validation splits:
1. Split 2 (Short-Term): Validates responsiveness to recent shifts in the last 6 months.
2. Split 3 (Full History Stress Test): Uses almost all available data to test stability.
This multi-split architecture ensures that the recommended model is not just a "one-hit wonder" but is structurally sound across different regimes.
"""
pdf.add_chapter("2. Evaluation Framework", period_text)

# --- 4. DATA PREPARATION ---
feat_text = """
3. Feature Engineering & Data Preparation

Traditional statistical models (like ARIMA) consume raw data, but modern Machine Learning models require "Feature Engineering"—the art of transforming raw time-series data into informative predictors.

For our Supervised Learning models (specifically XGBoost), we engineered the following features from the raw demand signal:

1. Lag Features (Autoregression):
   We created "Lag 1" (demand t-1 month ago) and "Lag 12" (demand t-12 months ago). 
   - Rationale: Lag 1 captures immediate momentum (if demand was high yesterday, it's likely high today). Lag 12 captures seasonality (if demand was high last January, it's likely high this January).

2. Temporal Features (Cyclical encoding):
   We extracted the "Month" (1-12) from the timestamp.
   - Rationale: This allows the model to learn calendar-specific effects, such as fiscal year-end pushes or holiday-driven slumps, without needing explicit holiday data files.

3. Weather/Seasonal Flags (Monsoon Engineering):
   We explicitly engineered binary boolean flags for specific Indian seasons to test if humidity/rainfall impacts demand:
   - `is_monsoon`: Flagged 1 for July and August (Peak Monsoon).
   - `is_pre_monsoon`: Flagged 1 for May (High heat/humidity onset).
   - Rationale: Certain parts may experience higher failure rates due to environmental conditions (rust, electrical shorting) during these specific windows.

4. Scaling and Normalization:
   For Neural Networks (N-HiTS), raw demand values can vary wildly (from 0 to 10,000). We applied standard scaling (Mean=0, Variance=1) to stabilize the gradient descent process during training, ensuring the network converges efficiently.
"""
pdf.add_chapter("3. Feature Engineering Strategy", feat_text)

# --- 5. ALGORITHMS ---
models_overview = """
4. Forecasting Algorithms: Selection & Rationale

We deliberately selected a "Council of Models"—a diverse set of six algorithms ranging from classical statistics to deep learning. This diversity is our primary defense against model drift.
"""
pdf.add_chapter("4. Algorithms & Model Architecture", models_overview)

models_text_1 = """
4.1 Exponential Smoothing (ETS / Holt-Winters)
- Overview: A robust statistical baseline that decomposes the series into Level, Trend, and Seasonality.
- Strengths: Extremely interpretable and theoretically sound for clearly seasonal data. Efficient on small datasets.
- Limitations: Struggles with complex, non-linear patterns or multiple seasonalities.
- Application: We used the Additive Holt-Winters method, optimizing the smoothing parameters (Alpha, Beta, Gamma) via AIC minimization.

4.2 SARIMA (Seasonal ARIMA)
- Overview: Captures autocorrelations and moving averages in the residuals.
- Strengths: Excellent at modeling the internal structure of the series and handling non-stationarity via differencing.
- Limitations: Computationally expensive to tune; degrades if the history is too short.
- Application: Configured as (1,1,1)(1,1,0)[12] to explicitly model the monthly dependency structure.

4.3 Prophet (Meta)
- Overview: A generalized additive model tailored for business time series with strong seasonal effects.
- Strengths: Handles missing data and outliers gracefully. We customized it with an Indian Holiday Calendar to capture festival impacts.
- Limitations: Can be slow to fit; sometimes over-smooths sharp spikes.
- Application: Used with 'yearly_seasonality=True' and custom India-specific holiday regressors.
"""
pdf.add_chapter("4.1 Statistical Models", models_text_1)

models_text_2 = """
4.4 XGBoost (Gradient Boosting)
- Overview: A powerful ensemble decision-tree algorithm.
- Strengths: Captures non-linear interactions between lags (e.g., "If last month was high AND it is December, then demand drops"). Very fast and high performance.
- Limitations: Cannot extrapolate trends (it predicts within the range of values it has seen). Requires careful feature engineering.
- Application: Trained on the engineered Lag-1 and Lag-12 features to predict future demand recursively.

4.5 N-HiTS (Neural Hierarchical Interpolation)
- Overview: A cutting-edge Deep Learning architecture (2022) that uses hierarchical blocks to model different frequencies (trends vs noise).
- Strengths: State-of-the-art accuracy on long-horizon forecasts. Captures global patterns that local statistical models miss.
- Limitations: Data hungry; computationally intensive (requires more CPU/GPU time).
- Application: Implemented via Darts with 3 stacks and 50 training epochs.

4.6 Weighted Ensemble
- Overview: Combines predictions from SARIMA, Prophet, and XGBoost using a weighted average.
- Strengths: Reduces variance and risk. If one model fails (e.g., Prophet over-predicts), the others (XGBoost) can correct it. "The wisdom of the crowds."
- Limitations: Complexity; difficult to interpret the "why" of a specific prediction.
- Application: Weights were dynamically assigned based on the inverse of the validation RMSE errors.
"""
pdf.add_chapter("4.2 Machine Learning & Ensembles", models_text_2)

# --- 6. SELECTION CRITERIA ---
selection_text = """
5. Optimal Model Selection: The Composite Score

Selecting the "best" model is a nuanced decision. A model with the lowest error might be unstable (high variance) or biased (consistently under-predicting). To solve this, we engineered a Composite Score.

5.1 The Metric Trinity
We calculate three key metrics for every model run:
1. MAPE (0.7 Weight): Measures accuracy. High weight because business stakeholders think in percentages.
2. RMSE (0.2 Weight): Measures stability. Penalizes large distinctive errors that could cause stockouts.
3. Bias (0.1 Weight): Measures direction. We prefer models with Bias near zero to avoid systematic inventory accumulation or loss.

5.2 The Scoring Algorithm
For each Part/Location, we normalize these metrics and compute:
   Score = (0.7 * Norm_MAPE) + (0.2 * Norm_RMSE) + (0.1 * Norm_Bias)

The model with the lowest Score is crowned the "Global Winner."

5.3 The Overfitting Guardrail
Before accepting a winner, we analyze the gap between Training MAPE and Testing MAPE.
- If Training Error is very low (1%) but Testing Error is high (20%), the model is Overfitting.
- If both are high, it is Underfitting.
- We prioritize models where the Training and Testing errors are comparable, indicating that the model has truly learned the pattern and will generalize well to 2025.
"""
pdf.add_chapter("5. Analysis & Decision Logic", selection_text)

# --- 7. IMPLEMENTATION ---
impl_text = """
6. Technical Implementation Details

The entire forecasting pipeline was synthesized into a seamless digital product.

1. Python Backend: The core logic uses `pandas` for data manipulation and `statsmodels`/`darts` for modeling.
2. Interactive Dashboard (Streamlit): A user-friendly web interface allows stakeholders to visualize trends, toggle between models, and view the "Best Fit" recommendations without touching code.
3. Deployment Pipeline (GitHub):
   - We established a CI/CD flow where local changes (verified on localhost) are pushed to GitHub.
   - Streamlit Cloud automatically builds the app from the `requirements.txt` file.
   - A custom "Deploy" button was built into the local dashboard to simplify this process.
4. Security: A simplified authentication gate ensures only authorized users (via Google Email) can access the sensitive forecasting data.
"""
pdf.add_chapter("6. Deployment Architecture", impl_text)

# --- 8. APPENDIX ---
pdf.add_chapter("Appendix: Source Code", "The following pages contain the complete, production-grade source code used in this project.")

files_to_print = [
    ('final_dashboard.py', 'Streamlit Dashboard Application'),
    ('generate_dashboard_data.py', 'Data Pipeline & Model Training Execution'),
    ('generate_future_forecast.py', '2025 Future Forecast Generator'),
    ('extract_history.py', 'Historical Data Extraction Utility'),
]

for fname, desc in files_to_print:
    if os.path.exists(fname):
        pdf.add_code_chapter(f"{fname} - {desc}", fname)
    else:
        pdf.add_chapter(f"Missing File: {fname}", "File not found.")

output_filename = "Demand_Forecasting_Detailed_Analysis.pdf"
pdf.output(output_filename, 'F')
print(f"Report generated: {output_filename}")
