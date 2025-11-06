import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Acceleration Component
    close = df['close']
    
    # Calculate price changes
    price_change_1 = close - close.shift(1)
    price_change_2 = close.shift(1) - close.shift(2)
    
    # Price acceleration: (Close_t - Close_t-1) - (Close_t-1 - Close_t-2)
    price_acceleration = price_change_1 - price_change_2
    
    # Calculate average absolute price change over 10 days
    avg_price_change_10d = price_change_1.abs().rolling(window=10, min_periods=1).mean()
    
    # Compute acceleration decay factor: exp(-|Price Acceleration|/Avg_Price_Change_10d)
    decay_weight = np.exp(-price_acceleration.abs() / avg_price_change_10d.replace(0, np.nan))
    decayed_acceleration = price_acceleration * decay_weight
    
    # Volume-Price Divergence
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Calculate normalized volume change: (Volume_t - Volume_t-1) / (High_t - Low_t)
    volume_change = volume - volume.shift(1)
    price_range = high - low
    normalized_volume_change = volume_change / price_range.replace(0, np.nan)
    
    # Calculate Price-Volume correlation over past 8 days
    correlation_window = 8
    price_volume_corr = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < correlation_window - 1:
            price_volume_corr.iloc[i] = 0
        else:
            start_idx = i - correlation_window + 1
            end_idx = i + 1
            window_volume = volume.iloc[start_idx:end_idx]
            window_close = close.iloc[start_idx:end_idx]
            corr = window_volume.corr(window_close)
            price_volume_corr.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Multiply Normalized Volume Change by Inverse Correlation
    inverse_correlation = 1 - price_volume_corr
    divergence_signal = normalized_volume_change * inverse_correlation
    
    # Combined Alpha Factor
    # Multiply Decayed Acceleration by Divergence Signal
    combined_factor = decayed_acceleration * divergence_signal
    
    # Apply Volume Confirmation: Multiply by sign(Volume_t - Volume_t-1)
    volume_confirmation = np.sign(volume_change)
    combined_factor = combined_factor * volume_confirmation
    
    # Scale by Recent Volatility: Divide by (High_t-2 - Low_t-2)
    recent_volatility = (high.shift(2) - low.shift(2)).replace(0, np.nan)
    final_factor = combined_factor / recent_volatility
    
    return final_factor
