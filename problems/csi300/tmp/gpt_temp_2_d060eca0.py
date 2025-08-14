import pandas as pd
def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor based on the given thought tree.
    
    Parameters:
    df (pd.DataFrame): A DataFrame indexed by (date) with columns [open, high, low, close, amount, volume].
    
    Returns:
    pd.Series: A Series indexed by (date) representing the factor values.
    """
    
    # Momentum Indicator
    sma = df['close'].rolling(window=20).mean()
    momentum = df['close'] - sma
    
    # Volatility Indicator
    daily_returns = df['close'].pct_change()
