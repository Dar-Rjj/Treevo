import pandas as pd

def heuristics_v2(df):
    # Calculate the rate of change (ROC) for 10-day and 30-day periods
    roc_10 = df['close'].pct_change(periods=10).fillna(0)
    roc_30 = df['close'].pct_change(periods=30).fillna(0)

    # Calculate the true range
    df['true_range'] = df[['high', 'low']].diff(axis=1).iloc[:, -1].abs()
    df['true_range'] = df[['true_range', (df['high'] - df['close']).abs(), (df['low'] - df['close']).abs()]].max(axis=1)

    # Calculate the average true range (ATR) over a 14-day period
    atr = df['true_range'].rolling(window=14).mean()

    # Generate the heuristic score as a weighted sum of ROCs and ATR
    heuristics_matrix = 0.6 * roc_10 + 0.4 * roc_30 - atr

    return heuristics_matrix
