import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-5 day momentum acceleration with volume-amount geometric alignment, normalized by 10-day volatility
    # Uses exponential decay for recent data emphasis
    
    # 3-day momentum
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # 5-day momentum  
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum acceleration (3-day vs 5-day)
    mom_accel = mom_3 - mom_5
    
    # Volume-amount geometric alignment (geometric mean of normalized volume and amount)
    vol_norm = df['volume'] / df['volume'].rolling(window=5).mean()
    amt_norm = df['amount'] / df['amount'].rolling(window=5).mean()
    vol_amt_align = (vol_norm * amt_norm) ** 0.5
    
    # Combine momentum acceleration with volume-amount alignment
    raw_factor = mom_accel * vol_amt_align
    
    # 10-day volatility (standard deviation of returns)
    returns = df['close'].pct_change()
    vol_10 = returns.rolling(window=10).std()
    
    # Apply exponential decay weights (0.9^lag) for recent emphasis
    weights = pd.Series([0.9**i for i in range(10)], index=range(10))
    vol_10_weighted = returns.rolling(window=10).apply(
        lambda x: (x * weights[:len(x)]).std(), raw=True
    )
    
    # Normalize by volatility with exponential decay
    factor = raw_factor / (vol_10_weighted + 1e-7)
    
    return factor
