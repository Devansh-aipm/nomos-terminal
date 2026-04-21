# Your Streamlit app code

import yfinance as yf

# Previous code...

df = yf.download(ticker, period="1y", auto_adjust=True)  # Removed multi_level_download parameter

# Continue with your Streamlit app...