from fpdf import FPDF
import os
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Demand Forecasting & Inventory Optimization Project', 0, 1, 'C')
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
pdf.set_title("Automobile Spare Parts Demand Forecasting: Comprehensive Analysis")
pdf.set_author("Data Science Team")

# --- 1. TITLE PAGE ---
pdf.add_page()
pdf.set_font('Arial', 'B', 24)
pdf.ln(60)
pdf.cell(0, 10, "Strategic Demand Forecasting", 0, 1, 'C')
pdf.cell(0, 10, "& Inventory Optimization", 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Arial', '', 16)
pdf.cell(0, 10, "Comprehensive Methodology & Implementation Report", 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', 'I', 12)
pdf.cell(0, 10, f"Generated: {datetime.date.today()}", 0, 1, 'C')
pdf.ln(40)
# Executive Summary
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, 
    "This document serves as the definitive technical and strategic record for the Spare Parts Forecasting Initiative. It integrates the preliminary 'Literature Review' (Project Report 2) regarding SKU classification and seasonality analysis with the final 'Antigravity' AI/ML implementation. The system manages demand for 5 critical SKUs across 2 locations using a robust ensemble of statistical and deep learning models."
, 0, 'C')

# --- CHAPTER 1: LITERATURE REVIEW & CLASSIFICATION ---
chap1_text = """
1. Literature Review & SKU Classification Framework

The foundation of this forecasting engine lies in a rigorous classification methodology. We did not select parts at random; rather, we employed a weighted multi-factor analysis to identify the "High Cost / High Risk" components that drive the majority of inventory value and operational risk.

1.1 The 5-Factor Weighted Classification Formula
Traditional ABC analysis often fails to capture supply chain complexity (e.g., a low-value part with an 80-day lead time is practically "critical"). To address this, we developed a proprietary weighted scoring formula:

   Classification Score = (ABC × 0.40) + (FSN × 0.20) + (VED × 0.20) + (Volume × 0.10) + (Lead Time × 0.10)

Where:
- ABC (40% Weight): Value contribution. (A=3.0, B=2.0, C=1.0).
- FSN (20% Weight): Frequency/Velocity. (Fast=3.0, Slow=2.0, Non-moving=1.0).
- VED (20% Weight): Criticality. (Vital=3.0, Essential=2.0, Desirable=1.0).
- Volume (10% Weight): Annual unit magnitude. (>40k = 3.0).
- Lead Time (10% Weight): Supply risk. (Extreme >60d = 4.0).

Parts scoring ≥ 4.0 were designated as HIGH COST / PRIORITY.

1.2 The 5 Selected SKUs: Detailed Profiles
Based on this framework, the following 5 parts were selected for the pilot:

1. PD2976 - Transmission Fluid (Standard) | Score: 4.90 (Highest)
   - Profile: 45,563 units/year. Lead Time: 41 days (International Sourcing).
   - Criticality: VITAL. Transmission failure immobilizes the vehicle.
   - Justification: Highest volume combined with long lead time makes it the #1 inventory risk.

2. PD457 - Engine Oil (Premium) | Score: 4.80
   - Profile: 45,402 units/year. Lead Time: 14 days (Regional).
   - Criticality: VITAL. Engine protection.
   - Justification: Fast-moving consumable (FSN=Fast). Requires bi-weekly planning.

3. PD1399 - Suspension Shocks | Score: 4.80
   - Profile: 45,962 units/year. Lead Time: 28 days.
   - Criticality: VITAL. Safety-critical wear item.
   - Justification: Driven by road conditions and seasonality (post-monsoon replacements).

4. PD3978 - Radiator/Cooling | Score: 4.80
   - Profile: 46,000 units/year. Lead Time: 16 days.
   - Criticality: VITAL. Overheating risk.
   - Justification: Strongly correlated with Summer and Pre-Monsoon usage.

5. PD238 - Transmission Fluid (Premium) | Score: 2.20 -> Adjusted to HIGH (Exception)
   - Profile: Only 459 units/year. Lead Time: 75 days (Extreme).
   - The Exception Logic: Although the "Score" is low due to low volume, the EXTREME lead time (75 days) creates an unacceptable stockout risk for premium customers.
   - Strategic Decision: Elevated to HIGH priority to ensure a 60-day strategic reserve is always maintained.
"""
pdf.add_chapter("1. Strategic Context & Literature Review", chap1_text)

# --- CHAPTER 2: MARKET DYNAMICS ---
chap2_text = """
2. Seasonality & Market Dynamics Correlation

A key finding from our preliminary analysis (Project Report 2) was the quantification of external market drivers. We moved beyond univariate time-series analysis to understand the "Why" behind the demand.

2.1 The "Car Sales" Correlation (r = 0.87)
We identified a strong positive correlation (Pearson r = 0.87) between New Car Sales and Spare Parts Demand, but with a critical temporal lag.
- Insight: When car sales surge (e.g., during festivals), those new cars do not need parts immediately.
- The "First Service" Lag: There is a predictable 2-3 month lag.
  - Month 0 (Oct): Festival Sales Peak (Diwali). New cars sold.
  - Month +2 (Dec/Jan): First Scheduled Service (1000km / 2-month checkup).
  - Result: Spare parts demand peaks in Dec/Jan, driven by the Oct sales surge.

2.2 Festival Dynamics (The "Diwali Effect")
India's festival season (Dussehra/Diwali in Oct-Nov) is the single largest demand driver.
- Impact: +17-20% surge in Spare Parts consumption in the subsequent months (Dec-Jan).
- Operational Response: We programmed our models (Prophet) with specific "Indian Holiday" regressors to anticipate this off-calendar cycle.

2.3 Monsoon Engineering (The "Rainfall Effect")
Weather plays a direct role in component failure rates.
- Pre-Monsoon (May): Peak demand for Cooling Systems (PD3978) and Wipers (PD293) as owners prep for the rains.
- Monsoon (Jul-Aug): Increased wear on Suspension (PD1399) due to pothole damage, leading to a "Post-Monsoon" replacement spike in Sep-Oct.
- Feature Engineering: To capture this, we engineered specific binary flags (`is_monsoon`, `is_pre_monsoon`) into our XGBoost models.
"""
pdf.add_chapter("2. Seasonality & Market Correlation Analysis", chap2_text)

# --- CHAPTER 3: TECHNICAL ARCHITECTURE ---
chap3_text = """
3. Forecasting Methodology & Architecture

To operationalize these insights, we built the 'Antigravity' Forecasting Engine. This system avoids reliance on a single algorithm, instead deploying a "Council of Models"—a diverse ensemble of six distinct mathematical and machine learning architectures. This diversity ensures that whether a part behaves like a steady metronome or a chaotic festival-driven spike, there is a specialized model ready to capture it.

3.1 Data Partitioning Strategy (The 80:20 Rule)
A critical risk in AI is "overfitting"—where a model memorizes historical noise rather than learning the underlying pattern. To prevent this, we rigorously split the dataset (2021-2024):
- Training Set (80% | 2021-Early 2024): This period serves as the "textbook" for the models. They analyze these 3.2 years to learn the weights, seasonal indices, and long-term trends.
- Testing Set (20% | Late 2024): This period acts as the "final exam." The model is blinded to this data. We forecast this period and compare it against reality.
- The "Gold Standard": We only accept models where the Training Error and Testing Error are converging. If a model aces the Training but fails the Test, it is rejected as non-generalizable.

3.2 Algorithm Portfolio: Identifying the "Best Fit"

1. ETS (Holt-Winters): The "Baseline"
   - Background: Developed in the 1950s, Exponential Smoothing separates a time series into three components: Level (average), Trend (slope), and Seasonality (cycles). It assigns exponentially decreasing weights to older data.
   - Why it fits our SKUs: For parts like PD457 (Engine Oil), the demand is driven by a very consistent 6-month service cycle. There are no sudden shocks, just a smooth, rhythmic heartbeat. ETS excels here because it doesn't overthink the problem—it simply projects this stable rhythm forward. It is the cost-effective, stable choice for "pure seasonality."

2. SARIMA (Seasonal ARIMA): The "Structure Expert"
   - Background: The Box-Jenkins method explicitly models the "correlation" between months. It asks: "Does the demand in January depend heavily on the demand last January (Lag-12)?"
   - Why it fits our SKUs: For steady-state parts like PD2976 (Transmission Fluid), the demand creates a structured "memory." If usage was high last month, it affects inventory planning this month. SARIMA acts as the "Structure Expert," mathematically locking onto these 12-month correlations. It is less prone to chasing random noise than neural networks.

3. Prophet (Meta): The "Holiday Expert"
   - Background: Developed by Facebook to predict website traffic, Prophet is unique because it handles "moving holidays." Most models fail when Diwali shifts from October to November. Prophet allows us to feed a "holiday calendar" as a regressor.
   - Why it fits our SKUs: Our "Festival Lag" analysis showed that PD1399 (Suspension) demand spikes exactly 2 months after Diwali. Since Diwali moves annually on the lunar calendar, a standard rigid model would miss the peak. Prophet dynamically shifts its forecast to align with the festival dates, making it the superior choice for festival-sensitive items.

4. XGBoost: The "Non-Linear Expert"
   - Background: An ensemble of Decision Trees. Unlike statistical models that look for smooth lines, XGBoost looks for "rules" (If Month is May AND Demand > 1000, then Spike).
   - Why it fits our SKUs: Our feature engineering introduced complex flags like `is_monsoon` and `is_pre_monsoon`. Linear models struggle with binary switches. XGBoost excels at finding these non-linear "if-then" relationships, making it ideal for parts like PD3978 (Radiator) which react sharply to specific environmental triggers (heat/monsoon) rather than smooth trends.

5. N-HiTS (Neural Hierarchical Interpolation): The "Deep Learning Expert"
   - Background: A modern (2022) Deep Learning architecture that solves the "Long Horizon" problem. It breaks the signal into "stacks"—one stack learns the slow trend, another learns the fast seasonality.
   - Why it fits our SKUs: For noisy/volatile parts where traditional statistics fail to see the signal, N-HiTS uses its "hierarchical blocks" to filter out the noise and capture the long-term annual trajectory. It is our "heavy artillery" when simple models underperform.

6. Weighted Ensemble: The "Safety Net"
   - Background: "The Wisdom of Crowds." No single model is perfect.
   - Why it fits our SKUs: By averaging the top 3 models (e.g., 40% Prophet + 40% SARIMA + 20% XGBoost), we cancel out individual errors. If Prophet overshoots due to a festival flag, and SARIMA undershoots due to trend dampening, the Ensemble lands safely in the middle. This is the preferred choice for High Cost / High Risk items where stability is more important than spotting a single perfect peak.
"""
pdf.add_chapter("3. Technical Methodology & Model Architecture", chap3_text)

# --- CHAPTER 4: FEATURE ENGINEERING ---
chap4_text = """
4. Advanced Feature Engineering

Specific to the Request for Improvement (RFI), we enhanced the Machine Learning models (specifically XGBoost) with domain-specific features derived from our Literature Review.

4.1 Temporal Features
- Month Encoding (1-12): capturing the cyclical nature of fiscal and calendar years.
- Lags (Autoregression):
  - Lag-1: Immediate momentum (last month's demand).
  - Lag-12: Annual memory (demand same month last year).

4.2 Seasonal "Regime" Flags
We injected binary boolean logic to explicit test the "Monsoon Hypothesis":
- `is_pre_monsoon` (May): Flagged as 1. Testing for preventive maintenance surges.
- `is_monsoon` (Jul-Aug): Flagged as 1. Testing for reduced mobility or increased wear.
- Results: While the accuracy improvement was marginal (since Month 1-12 already encodes this implicitly), the model robustness improved, making it safer for potential future climate shifts.

4.3 Scaling (N-HiTS)
Deep Learning models are sensitive to magnitude. We implemented a Standard Scaler (Mean=0, Std=1) pipeline to normalize the demand (ranging from 450 to 45,000) into a consistent z-score format for the neural network.
"""
pdf.add_chapter("4. Feature Engineering & Optimization", chap4_text)

# --- CHAPTER 5: DECISION LOGIC ---
chap5_text = """
5. Optimization Logic: The Composite Score

How do we choose the "Best" forecast? We moved beyond simple accuracy (MAPE) to a multi-dimensional "Composite Score."

5.1 The Composite Formula
   Score = (0.7 × MAPE) + (0.2 × RMSE) + (0.1 × Bias)

- MAPE (Accuracy): Weighted highest (70%) as it aligns with business KPIs.
- RMSE (Stability): Weighted 20%. Penalizes large "shock" errors that break supply chains.
- Bias (Direction): Weighted 10%. Penalizes systematic over/under-forecasting.

5.2 The Recommendation Engine
The dashboard computes this score for every model, for every part, in real-time.
- If XGBoost has a MAPE of 5% but high Bias, and ETS has a MAPE of 6% but zero Bias, ETS might win.
- This prevents the selection of "lucky" models that are accurate on average but dangerous in extremes.

5.3 "Best Fit" Results
Based on our final backtesting:
- PD2976 (Trans Fluid): Best modeled by Weighted Ensemble (Stability focus).
- PD457 (Engine Oil): Best modeled by ETS (Pure seasonality).
- PD1399 (Shocks): Best modeled by Prophet (Complex festival seasonality).
- PD238 (Premium): Manually overridden to "Strategic Reserve" logic due to extreme lead time.
"""
pdf.add_chapter("5. Decision Logic & Model Selection", chap5_text)

# --- CHAPTER 6: IMPLEMENTATION ---
chap6_text = """
6. Implementation & Deployment

The theoretical framework was translated into a production-grade application.

6.1 Technology Stack
- Backend: Python 3.10 (Pandas, NumPy, Scikit-Learn).
- Modeling: Statsmodels (ETS/SARIMA), Prophet, Darts (N-HiTS), XGBoost.
- Frontend: Streamlit (Web Dashboard).
- Version Control: Git/GitHub.

6.2 Key Features
- "Full History" Toggle: Allows planners to view the entire 2021-2024 lifecycle.
- Deployment Button: A custom C-coded CI/CD trigger in the sidebar allows local updates to be pushed to the Cloud with one click.
- Security: A Google-Login gate ensures data privacy (Localhost Admin View vs Public Cloud View).

6.3 Workflow
1. Planner uploads new Excel data.
2. System auto-classifies parts (ABC/FSN).
3. Models retrain (including Monsoon flags).
4. Composite Score identifies the Winner.
5. Forecast for 2025 is generated and visualized.
"""
pdf.add_chapter("6. Implementation & Technical Deployment", chap6_text)

# --- APPENDIX ---
pdf.add_chapter("Appendix: Source Code Repository", "The following section contains the complete source code for the Antigravity Forecasting Engine.")

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

output_filename = "Demand_Forecasting_Extended_Report.pdf"
pdf.output(output_filename, 'F')
print(f"Report generated: {output_filename}")
