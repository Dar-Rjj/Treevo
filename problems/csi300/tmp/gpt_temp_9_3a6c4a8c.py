import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Volatility factor: Standard deviation of closing prices
    volatility = df['close'].rolling(window=5).std()

    # VWAP (Volume-Weighted Average Price)
    vwap = (df['amount'] / df['volume']).rolling(window=5).mean()

    # Momentum factor: 5-day return
