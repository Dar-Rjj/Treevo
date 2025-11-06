import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-5 day momentum acceleration blended with volume-amount geometric alignment
    # Normalized by 10-day volatility with exponential decay for recent emphasis
    
    # 3-day momentum
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # 5-day momentum  
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum acceleration (3-day vs 5-day)
    mom_accel = mom_3d - mom_5d
    
    # Volume-amount geometric alignment (geometric mean of normalized ratios)
    volume_norm = df['volume'] / df['volume'].rolling(window=5).mean()
    amount_norm = df['amount'] / df['amount'].rolling(window=5).mean()
    vol_amount_align = (volume_norm * amount_norm) ** 0.5
    
    # Blend momentum acceleration with volume-amount alignment
    raw_factor = mom_accel * vol_amount_align
    
    # 10-day volatility (using close-to-close returns)
    returns = df['close'].pct_change()
    volatility_10d = returns.rolling(window=10).std()
    
    # Exponential decay weights (recent emphasis: 0.9^lag)
    weights = pd.Series([0.9**i for i in range(10)], index=range(10))
    weighted_vol = returns.rolling(window=10).apply(lambda x: (x * weights[:len(x)]).sum() / weights[:len(x)].sum(), raw=True)
    
    # Normalize by volatility with decay adjustment
    factor = raw_factor / (weighted_vol.abs() + 1e-7)
    
    return factor
