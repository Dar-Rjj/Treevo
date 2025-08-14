def heuristics_v2(df):
    # Calculate the 14-day Average True Range (ATR) as a measure of volatility
    tr = df['high'] - df['low']
    atr = tr.rolling(window=14).mean()
    
    # Calculate the 14-day cumulative sum of the difference in daily trading volumes
    volume_diff_cumsum = df['volume'].diff().rolling(window=14).sum()
    
    # Calculate the 14-day Average Directional Index (ADX)
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    di_plus = 100 * (dm_plus.rolling(window=14).sum() / atr)
    di_minus = 100 * (dm_minus.rolling(window=14).sum() / atr)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14).mean()
    
    # Combine ATR, cumulative volume difference, and ADX
    heuristics_matrix = (atr * volume_diff_cumsum) * adx
    
    return heuristics_matrix
