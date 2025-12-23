# Dashboard Deployment Guide 🚀

## 1. Setup GitHub Repository
1.  Create a new repository on GitHub (e.g., `spare-parts-dashboard`).
2.  Do **not** maximize "Initialize this repository with a README".
3.  Push this folder to the new repository using the terminal or GitHub Desktop:

```bash
git init
git add .
git commit -m "Initial commit of dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## 2. Deploy to Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Log in with your GitHub account.
3.  Click **"New app"**.
4.  Select your new repository (`spare-parts-dashboard`).
5.  **Main file path:** `final_dashboard.py`
6.  Click **"Deploy!"**.

## 3. Data & Secrets
- The `Dashboard_Database.csv` and `Future_Forecast_Database.csv` files are included in the git commit, so the dashboard will work immediately with the pre-generated data.
- **Note:** The raw Excel file `Spare-Part-Data-With-Summary.xlsx` is excluded by `.gitignore` to protect sensitive data. The cloud dashboard is a **viewer** only; you must regenerate data locally and push the updated CSVs to update the cloud view.

## 4. Dependencies
- `requirements.txt` is configured for the **Lite Dashboard Viewer** (Streamlit, Pandas, Plotly) to ensure fast deployment.
