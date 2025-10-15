import pandas as pd
import yfinance as yf

def download_data(symbol, start, end):
    """Download OHLCV data using yfinance."""
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[col for col in expected_cols if col in df.columns]].copy()
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    """Add technical indicators: returns, SMA, RSI."""
    df = df.copy()
    df['return'] = df['Close'].pct_change().fillna(0)
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / (roll_down + 1e-9)
    df['rsi'] = 100.0 - (100.0 / (1.0 + RS))

    # Fill NaN values
    df['sma_5'] = df['sma_5'].bfill()
    df['sma_20'] = df['sma_20'].bfill()
    df['rsi'] = df['rsi'].fillna(50)

    return df
