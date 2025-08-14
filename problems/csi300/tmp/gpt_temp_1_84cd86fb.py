import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Volume Weighted Average Price (VWAP)
    typical_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Combine Intraday Range and VWAP
    diff_high_vwap = df['high'] - vwap
    diff_vwap_low = vwap - df['low']
    combined_range_vwap = diff_high_vwap + diff_vwap_low
    
    # Calculate Intraday Price Changes
    log_open_close_change = (df['close'] / df['open']).apply(lambda x: math.log(x))
    log_high_low_change = (df['high'] / df['low']).apply(lambda x: math.log(x))
    
    # Aggregate Intraday Price Changes
    aggregated_price_changes = log_open_close_change + log_high_low_change
    
    # Calculate Volume Synchrony
    volume_ma = df['volume'].rolling(window=n).mean()
    daily_volume_deviation = df['volume'] - volume_ma
    
    # Integrate Combined Intraday Range-VWAP and Aggregated Price Changes with Volume Synchrony
    integrated_factor = (combined_range_vwap * aggregated_price_changes) * daily_volume_deviation
    
    # Calculate High-Low Price Range Momentum
    close_momentum = df['close'].diff(periods=n)
    
    # Calculate Volume Weighted Average True Range (VWATR)
    true_range = df[['high', 'low']].max(axis=1) - df[['high', 'low']].min(axis=1)
    vwatr = (true_range * df['volume']).rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
    
    # Adjust Momentum by VWATR
    adjusted_momentum = close_momentum / vwatr
    
    # Final Alpha Factor
    final_alpha = integrated_factor * adjusted_momentum
    
    # Apply a moving average to smooth the factor
    smoothed_alpha = final_alpha.rolling(window=n).mean()
    
    return smoothed_alpha
