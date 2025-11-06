import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-adjusted momentum with volume divergence
    # Captures sustainable price moves confirmed by volume divergence from recent patterns
    
    # 5-day price momentum (using proper lag to avoid lookahead)
    price_momentum = (df['close'].shift(1) - df['close'].shift(6)) / df['close'].shift(6)
    
    # Volume divergence: current volume vs 10-day median (more robust than mean)
    volume_divergence = df['volume'] / df['volume'].rolling(window=10).median()
    
    # Efficiency ratio: net price change vs total price movement
    price_range = (df['high'] - df['low']).abs()
    net_change = (df['close'] - df['open']).abs()
    efficiency_ratio = net_change / (price_range + 1e-7)
    
    # Volatility regime adjustment using rolling percentiles
    recent_volatility = (df['high'] - df['low']) / df['close']
    vol_regime = recent_volatility.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 - 1)
    
    # Composite factor: momentum amplified by volume divergence, 
    # filtered by efficiency and adjusted for volatility regime
    factor = price_momentum * volume_divergence * efficiency_ratio * vol_regime
    
    return factor
