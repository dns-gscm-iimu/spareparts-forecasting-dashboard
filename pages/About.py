import streamlit as st

st.set_page_config(page_title="About - automobile Spare Parts Forecasting", layout="wide")

# --- GLOBAL CSS (Canela Font) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');

h1, h2, h3, h4, h5, h6, div, p {
    font-family: 'Canela', 'Playfair Display', serif !important;
}
</style>
""", unsafe_allow_html=True)

# Vibrant About Section HTML
ABOUT_HTML = """
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 60px; border-radius: 20px; margin-top: 20px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(0, 242, 254, 0.3);">
<h1 style="margin-bottom: 30px; font-size: 48px; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); color:white; padding:0;">Tech Mahindra x IIM Udaipur</h1>
<div style="font-size: 28px; font-weight: 500; line-height: 1.6; margin-bottom: 30px;">Project made under the guidance of<br><b>Mr. Adarsh Uppoor (Tech Mahindra)</b> & <b>Prof. Rahul Pandey (IIM Udaipur)</b></div>
<div style="font-size: 24px; font-weight: 400; opacity: 0.9;">Made with Curiosity by <b>Deevyendu N Shukla</b> & <b>Vishesh Bhargava</b><br>IIM Udaipur GSCM 2025-26</div>
</div>
"""

st.markdown(ABOUT_HTML, unsafe_allow_html=True)
