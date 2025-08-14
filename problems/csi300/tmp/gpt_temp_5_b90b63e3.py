import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Volatility factor: Standard deviation of closing prices with dynamic window
    volatility = df['close'].rolling(window=df['close'].rolling(window=5).std().astype(int), min_periods=1).std()

    # VWAP (Volume-Weighted Average Price) with dynamic window
    vwap_window = df['volume'].rolling(window=5).mean().astype(int)
    vwap = (df['amount'] / df['volume']).rolling(window=vwap_window, min_periods=1).mean()

    # Momentum factor: 5-day return with dynamic window
