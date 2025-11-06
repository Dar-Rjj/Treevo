import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Timeframe Momentum Components
    # Short-Term Momentum (5-day)
    mom_5d = (data['close'].shift(1) / data['close'].shift(5) - 1)
    
    # Medium-Term Momentum (20-day)
    mom_20d = (data['close'].shift(1) / data['close'].shift(20) - 1)
    
    # Momentum Divergence Detection
    mom_divergence = mom_5d - mom_20d
    mom_divergence_squared = np.sign(mom_divergence) * (mom_divergence ** 2)
    
    # Calculate Volatility-Adjusted Components
    # Price Volatility Measure (10-day ATR)
    high_low_range = data['high'] - data['low']
    high_prev_close = np.abs(data['high'] - data['close'].shift(1))
    low_prev_close = np.abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr_10d = true_range.rolling(window=10, min_periods=5).mean()
    
    # Volatility-Adjusted Momentum Divergence
    vol_adj_mom_div = mom_divergence_squared / (atr_10d / data['close'].shift(1))
    
    # Recent Price Range Adjustment
    high_5d = data['high'].rolling(window=5, min_periods=3).max()
    low_5d = data['low'].rolling(window=5, min_periods=3).min()
    price_range_5d = (high_5d - low_5d) / data['close']
    vol_scaling = 1 / (price_range_5d + 1e-6)
    
    # Calculate Volume Acceleration Divergence
    # Volume Momentum Components
    vol_mom_5d = (data['volume'].shift(1) / data['volume'].shift(5) - 1)
    vol_mom_10d = (data['volume'].shift(1) / data['volume'].shift(10) - 1)
    
    # Volume Acceleration
    vol_accel = vol_mom_5d - vol_mom_10d
    
    # Volume Divergence Detection
    vol_divergence = vol_mom_5d - vol_mom_10d
    vol_divergence_cubed = np.sign(vol_divergence) * (np.abs(vol_divergence) ** (1/3))
    
    # Combine Components with Divergence Logic
    # Primary Divergence Signal
    primary_signal = vol_adj_mom_div * vol_divergence_cubed * np.sign(mom_5d)
    
    # Volume Confirmation
    vol_confirmation = np.sign(vol_accel) * np.log1p(np.abs(vol_accel))
    
    # Final Combination
    combined_signal = primary_signal * vol_confirmation * vol_scaling
    
    # Apply Adaptive Smoothing
    # Dynamic Smoothing Window based on volatility
    volatility_measure = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    smoothing_window = np.where(volatility_measure > volatility_measure.rolling(window=50).median(), 
                               5, 10)
    
    # Apply volatility-adaptive moving average
    alpha_factor = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= max(smoothing_window):
            window = int(smoothing_window[i])
            start_idx = max(0, i - window + 1)
            alpha_factor.iloc[i] = combined_signal.iloc[start_idx:i+1].mean()
        else:
            alpha_factor.iloc[i] = combined_signal.iloc[:i+1].mean()
    
    return alpha_factor
