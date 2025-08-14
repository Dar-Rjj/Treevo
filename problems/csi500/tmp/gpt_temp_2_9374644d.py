import pandas as pd
import pandas as pd

def heuristics_v2(df, N=10, M=20):
    # Calculate Intraday Volatility Expansion
    today_high_low_range = df['high'] - df['low']
    prev_day_high_low_range = (df['high'].shift(1) - df['low'].shift(1)).fillna(method='bfill')
    intraday_volatility_expansion = (today_high_low_range - prev_day_high_low_range) / prev_day_high_low_range
    
    # Volume Weighted Moving Average
    close_prices = df['close']
    volumes = df['volume']
    weighted_close_prices = close_prices.rolling(window=N).apply(lambda x: (x * volumes).sum(), raw=False)
    total_volumes = volumes.rolling(window=N).sum()
    vwma = weighted_close_prices / total_volumes
    
    # Price Momentum
    n_day_close_diff = close_prices - close_prices.shift(N)
    m_day_close_diff = close_prices - close_prices.shift(M)
    price_momentum = (n_day_close_diff + m_day_close_diff) / 2
    
    # Volume Surge
    n_day_volume_change = volumes - volumes.shift(N)
    m_day_volume_change = volumes - volumes.shift(M)
    volume_surge = (n_day_volume_change + m_day_volume_change) / 2
    volume_surge_adjusted = volume_surge.apply(lambda x: x * 0.75 if x < 0 else x)
    
    # Final Alpha Factor
    alpha_factor = (intraday_volatility_expansion + vwma + price_momentum + volume_surge_adjusted) / 4
    
    return alpha_factor
