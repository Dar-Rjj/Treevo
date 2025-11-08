import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel alpha factor: Clean Breakout Signal with Volume Confirmation and Volatility Scaling
    # Combines clear breakout identification, strong volume confirmation, and volatility adjustment
    # Interpretable as: Stocks making clean breakouts from recent ranges with strong volume support and low volatility tend to continue trending
    
    # Calculate clean breakout signal (distance from recent range midpoint)
    # More robust than simple high/low proximity, captures breakout direction clearly
    five_day_high = df['high'].rolling(window=5).max()
    five_day_low = df['low'].rolling(window=5).min()
    range_midpoint = (five_day_high + five_day_low) / 2
    clean_breakout = (df['close'] - range_midpoint) / (five_day_high - five_day_low + 1e-7)
    
    # Calculate strong volume confirmation (volume percentile in recent period)
    # Uses rolling quantile to identify unusually high volume more robustly
    volume_5d_quantile = df['volume'].rolling(window=5).apply(lambda x: (x[-1] > x[:-1]).mean())
    strong_volume_confirmation = volume_5d_quantile
    
    # Calculate volatility using normalized true range
    # More stable volatility measure for scaling breakout signals
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    normalized_volatility = true_range / df['close'].shift(1)
    volatility_5d = normalized_volatility.rolling(window=5).mean()
    
    # Combine factors: clean breakout amplified by strong volume confirmation
    # Volatility scaling penalizes high-volatility breakouts
    alpha_factor = clean_breakout * strong_volume_confirmation / (volatility_5d + 1e-7)
    
    return alpha_factor
