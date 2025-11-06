import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Volatility-Adjusted Cross-Timeframe Signals
    """
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Multi-Timeframe Price Momentum (Exponentially Smoothed)
    price_momentum_2d = close.diff(2)
    price_momentum_5d = close.diff(5)
    price_momentum_10d = close.diff(10)
    price_momentum_20d = close.diff(20)
    
    # Initialize smoothed price momentum
    smoothed_price_2d = pd.Series(index=df.index, dtype=float)
    smoothed_price_5d = pd.Series(index=df.index, dtype=float)
    smoothed_price_10d = pd.Series(index=df.index, dtype=float)
    smoothed_price_20d = pd.Series(index=df.index, dtype=float)
    
    # Calculate exponentially smoothed price momentum
    for i in range(len(df)):
        if i >= 20:
            if i == 20:
                smoothed_price_2d.iloc[i] = price_momentum_2d.iloc[i]
                smoothed_price_5d.iloc[i] = price_momentum_5d.iloc[i]
                smoothed_price_10d.iloc[i] = price_momentum_10d.iloc[i]
                smoothed_price_20d.iloc[i] = price_momentum_20d.iloc[i]
            else:
                smoothed_price_2d.iloc[i] = 0.3 * price_momentum_2d.iloc[i] + 0.7 * smoothed_price_2d.iloc[i-1]
                smoothed_price_5d.iloc[i] = 0.5 * price_momentum_5d.iloc[i] + 0.5 * smoothed_price_5d.iloc[i-1]
                smoothed_price_10d.iloc[i] = 0.7 * price_momentum_10d.iloc[i] + 0.3 * smoothed_price_10d.iloc[i-1]
                smoothed_price_20d.iloc[i] = 0.9 * price_momentum_20d.iloc[i] + 0.1 * smoothed_price_20d.iloc[i-1]
    
    # Multi-Timeframe Volume Momentum (Exponentially Smoothed)
    volume_momentum_2d = volume.diff(2)
    volume_momentum_5d = volume.diff(5)
    volume_momentum_10d = volume.diff(10)
    volume_momentum_20d = volume.diff(20)
    
    # Initialize smoothed volume momentum
    smoothed_volume_2d = pd.Series(index=df.index, dtype=float)
    smoothed_volume_5d = pd.Series(index=df.index, dtype=float)
    smoothed_volume_10d = pd.Series(index=df.index, dtype=float)
    smoothed_volume_20d = pd.Series(index=df.index, dtype=float)
    
    # Calculate exponentially smoothed volume momentum
    for i in range(len(df)):
        if i >= 20:
            if i == 20:
                smoothed_volume_2d.iloc[i] = volume_momentum_2d.iloc[i]
                smoothed_volume_5d.iloc[i] = volume_momentum_5d.iloc[i]
                smoothed_volume_10d.iloc[i] = volume_momentum_10d.iloc[i]
                smoothed_volume_20d.iloc[i] = volume_momentum_20d.iloc[i]
            else:
                smoothed_volume_2d.iloc[i] = 0.3 * volume_momentum_2d.iloc[i] + 0.7 * smoothed_volume_2d.iloc[i-1]
                smoothed_volume_5d.iloc[i] = 0.5 * volume_momentum_5d.iloc[i] + 0.5 * smoothed_volume_5d.iloc[i-1]
                smoothed_volume_10d.iloc[i] = 0.7 * volume_momentum_10d.iloc[i] + 0.3 * smoothed_volume_10d.iloc[i-1]
                smoothed_volume_20d.iloc[i] = 0.9 * volume_momentum_20d.iloc[i] + 0.1 * smoothed_volume_20d.iloc[i-1]
    
    # Dynamic Volatility Estimation
    # True Range Calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 5-day ATR (Exponentially Weighted)
    atr_5d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            if i == 5:
                atr_5d.iloc[i] = true_range.iloc[i-4:i+1].mean()
            else:
                atr_5d.iloc[i] = 0.6 * true_range.iloc[i] + 0.4 * atr_5d.iloc[i-1]
    
    # 10-day Return Standard Deviation (Exponentially Weighted)
    returns_10d = close.pct_change(periods=10)
    return_std_10d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            current_var = returns_10d.iloc[i-9:i+1].var()
            if i == 10:
                return_std_10d.iloc[i] = np.sqrt(current_var)
            else:
                return_std_10d.iloc[i] = np.sqrt(0.8 * current_var + 0.2 * return_std_10d.iloc[i-1]**2)
    
    # Volume Volatility Measures
    daily_volume_range = abs(volume - volume.shift(1))
    
    # 5-day Average Volume Range (Exponentially Weighted)
    avg_volume_range_5d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            if i == 5:
                avg_volume_range_5d.iloc[i] = daily_volume_range.iloc[i-4:i+1].mean()
            else:
                avg_volume_range_5d.iloc[i] = 0.6 * daily_volume_range.iloc[i] + 0.4 * avg_volume_range_5d.iloc[i-1]
    
    # Volume Change Standard Deviation (Exponentially Weighted)
    volume_changes = volume.pct_change(periods=10)
    volume_std_10d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            current_var = volume_changes.iloc[i-9:i+1].var()
            if i == 10:
                volume_std_10d.iloc[i] = np.sqrt(current_var)
            else:
                volume_std_10d.iloc[i] = np.sqrt(0.8 * current_var + 0.2 * volume_std_10d.iloc[i-1]**2)
    
    # Volatility-Adjusted Momentum Signals
    vol_adj_price_2d = smoothed_price_2d / (atr_5d + return_std_10d + 1e-8)
    vol_adj_price_5d = smoothed_price_5d / (atr_5d + return_std_10d + 1e-8)
    vol_adj_price_10d = smoothed_price_10d / (atr_5d + return_std_10d + 1e-8)
    vol_adj_price_20d = smoothed_price_20d / (atr_5d + return_std_10d + 1e-8)
    
    vol_adj_volume_2d = smoothed_volume_2d / (avg_volume_range_5d + volume_std_10d + 1e-8)
    vol_adj_volume_5d = smoothed_volume_5d / (avg_volume_range_5d + volume_std_10d + 1e-8)
    vol_adj_volume_10d = smoothed_volume_10d / (avg_volume_range_5d + volume_std_10d + 1e-8)
    vol_adj_volume_20d = smoothed_volume_20d / (avg_volume_range_5d + volume_std_10d + 1e-8)
    
    # Cross-Timeframe Price-Volume Divergence
    divergence_2d = vol_adj_price_2d - vol_adj_volume_2d
    divergence_5d = vol_adj_price_5d - vol_adj_volume_5d
    divergence_10d = vol_adj_price_10d - vol_adj_volume_10d
    divergence_20d = vol_adj_price_20d - vol_adj_volume_20d
    
    # Acceleration Divergence
    price_acceleration = (vol_adj_price_2d - vol_adj_price_5d) - (vol_adj_price_5d - vol_adj_price_10d)
    volume_acceleration = (vol_adj_volume_2d - vol_adj_volume_5d) - (vol_adj_volume_5d - vol_adj_volume_10d)
    
    # Cross-Timeframe Correlation Analysis
    corr_5d = pd.Series(index=df.index, dtype=float)
    corr_10d = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 20:
            # 5-day correlation
            if i >= 24:
                price_window_5d = price_momentum_5d.iloc[i-4:i+1]
                volume_window_5d = volume_momentum_5d.iloc[i-4:i+1]
                if len(price_window_5d) >= 3 and len(volume_window_5d) >= 3:
                    corr_5d.iloc[i] = price_window_5d.corr(volume_window_5d)
            
            # 10-day correlation
            if i >= 29:
                price_window_10d = price_momentum_10d.iloc[i-9:i+1]
                volume_window_10d = volume_momentum_10d.iloc[i-9:i+1]
                if len(price_window_10d) >= 5 and len(volume_window_10d) >= 5:
                    corr_10d.iloc[i] = price_window_10d.corr(volume_window_10d)
    
    correlation_divergence = corr_5d - corr_10d
    
    # Signal Integration Framework
    for i in range(len(df)):
        if i >= 29:  # Ensure sufficient data for all calculations
            # Timeframe Weighting Strategy
            weighted_divergence = (
                0.4 * divergence_2d.iloc[i] +  # Ultra-short weight
                0.3 * divergence_5d.iloc[i] +  # Short-term weight
                0.2 * divergence_10d.iloc[i] + # Medium-term weight
                0.1 * divergence_20d.iloc[i]   # Long-term weight
            )
            
            # Correlation-Based Signal Adjustment
            if not pd.isna(corr_5d.iloc[i]):
                correlation_factor = 1.0
                if abs(corr_5d.iloc[i]) < 0.2:  # Low correlation
                    correlation_factor = 1.5  # Amplify divergence signals
                elif abs(corr_5d.iloc[i]) > 0.6:  # High correlation
                    correlation_factor = 0.7  # Dampen divergence signals
                
                weighted_divergence *= correlation_factor
            
            # Acceleration Confirmation
            acceleration_multiplier = 1.0
            if not (pd.isna(price_acceleration.iloc[i]) or pd.isna(volume_acceleration.iloc[i])):
                if price_acceleration.iloc[i] > 0 and volume_acceleration.iloc[i] < 0:
                    acceleration_multiplier = 1.3  # Strong bullish
                elif price_acceleration.iloc[i] < 0 and volume_acceleration.iloc[i] > 0:
                    acceleration_multiplier = 0.7  # Strong bearish
                elif (price_acceleration.iloc[i] > 0 and volume_acceleration.iloc[i] > 0) or \
                     (price_acceleration.iloc[i] < 0 and volume_acceleration.iloc[i] < 0):
                    acceleration_multiplier = 1.1  # Trend continuation
            
            # Signal Quality Assessment
            volatility_threshold = atr_5d.iloc[i] > atr_5d.quantile(0.2)
            volume_momentum_check = abs(smoothed_volume_5d.iloc[i]) > abs(smoothed_volume_5d).quantile(0.3)
            
            # Cross-timeframe consistency check
            timeframe_alignment = (
                np.sign(divergence_2d.iloc[i]) == np.sign(divergence_5d.iloc[i]) or
                np.sign(divergence_5d.iloc[i]) == np.sign(divergence_10d.iloc[i])
            )
            
            # Final Alpha Factor Construction
            if volatility_threshold and volume_momentum_check and timeframe_alignment:
                final_signal = weighted_divergence * acceleration_multiplier
            else:
                final_signal = weighted_divergence * 0.5  # Reduce signal strength for poor quality
            
            # Add correlation divergence as additional signal component
            if not pd.isna(correlation_divergence.iloc[i]):
                final_signal += correlation_divergence.iloc[i] * 0.1
            
            result.iloc[i] = final_signal
    
    # Fill NaN values with 0 for early periods
    result = result.fillna(0)
    
    return result
