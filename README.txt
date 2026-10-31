QUANT EDGE - SERENDIB ENDOWMENT PROJECT

WHAT THIS PROJECT IS
This folder contains the Streamlit app for the Quant Edge competition. It uses past prices for SPY, EFA, EEM, AGG, GLD, XLE and XLK to show a sample portfolio and some risk measures.

HOW TO RUN IT
You need Python 3.10 or newer and an internet connection.

1. Open PowerShell in this folder.
2. Run these commands:

   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   streamlit run app.py

3. The app should open in your web browser. Use the sidebar to choose dates and settings.

DATA AND METHODS
The app downloads adjusted ETF prices from Yahoo Finance and the daily 3-month US Treasury yield (DGS3MO) from FRED. It calculates portfolio weights, historical performance, VaR and Expected Shortfall, an EVT tail estimate, backtests, crisis examples, and other measures shown in the app. Data providers can update past records, so note the date range and review the results before using them.

AI USE DISCLOSURE

AI was used to catch coding errors and ensure clean formatting.
