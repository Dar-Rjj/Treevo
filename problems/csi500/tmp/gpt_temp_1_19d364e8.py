import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum-Volume Acceleration Factor
    Combines intraday reversal signals with price and volume acceleration,
    adjusted for volatility regimes and price persistence patterns.
    """
    data = df.copy()
    
    # 1. Intraday Price Extremes
    intraday_high = data['high']
    intraday_low = data['low']
    
    # 2. Intraday Reversal Signals
    close_to_high_dist = intraday_high - data['close']
    close_to_low_dist = data['close'] - intraday_low
    
    # 3. Multi-period Price Acceleration
    mom_3d = data['close'] - data['close'].shift(3)
    mom_6d = data['close'] - data['close'].shift(6)
    price_acceleration = mom_3d - mom_6d
    
    # 4. Volume Acceleration
    vol_3d_avg = data['volume'].rolling(window=3, min_periods=1).mean()
    vol_6d_avg = data['volume'].rolling(window=6, min_periods=1).mean()
    volume_acceleration = vol_3d_avg - vol_6d_avg
    
    # 5. Regime Asymmetry
    daily_range = data['high'] - data['low']
    avg_10d_range = daily_range.rolling(window=10, min_periods=1).mean()
    median_20d_range = daily_range.rolling(window=20, min_periods=1).median()
    
    # Classify volatility regime (1 for high, 0 for low)
    high_vol_regime = (avg_10d_range > median_20d_range).astype(int)
    
    # 6. Combine Intraday Reversal with Acceleration
    high_vol_signal_high = close_to_high_dist * price_acceleration
    high_vol_signal_low = close_to_low_dist * price_acceleration
    
    low_vol_signal_high = close_to_high_dist * volume_acceleration
    low_vol_signal_low = close_to_low_dist * volume_acceleration
    
    # Apply regime-specific weighting
    combined_signal_high = (high_vol_regime * high_vol_signal_high + 
                           (1 - high_vol_regime) * low_vol_signal_high)
    combined_signal_low = (high_vol_regime * high_vol_signal_low + 
                          (1 - high_vol_regime) * low_vol_signal_low)
    
    # Net intraday reversal signal
    net_intraday_signal = combined_signal_high - combined_signal_low
    
    # 7. Price Persistence Adjustment
    close_returns = data['close'].pct_change()
    
    def count_consecutive_moves(returns_series, lookback=5):
        consecutive_counts = pd.Series(index=returns_series.index, dtype=float)
        
        for i in range(len(returns_series)):
            if i < lookback:
                consecutive_counts.iloc[i] = 0
                continue
                
            window = returns_series.iloc[i-lookback+1:i+1]
            if len(window) < lookback:
                consecutive_counts.iloc[i] = 0
                continue
                
            current_sign = np.sign(window.iloc[-1])
            count = 0
            
            for j in range(len(window)-1, -1, -1):
                if np.sign(window.iloc[j]) == current_sign and window.iloc[j] != 0:
                    count += 1
                else:
                    break
            
            consecutive_counts.iloc[i] = count
        
        return consecutive_counts
    
    consecutive_moves = count_consecutive_moves(close_returns, lookback=5)
    
    # Adjust signal strength based on persistence
    # Strong trends get amplified, potential reversals get dampened
    persistence_adjustment = np.where(
        consecutive_moves >= 3,  # Strong trend
        1.5,  # Amplify
        np.where(
            consecutive_moves <= 1,  # No clear trend
            0.8,  # Dampen
            1.0  # Neutral
        )
    )
    
    adjusted_signal = net_intraday_signal * persistence_adjustment
    
    # 8. Volatility Context and Final Factor
    # Calculate 15-day Average True Range (ATR)
    def calculate_atr(data, window=15):
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        return atr
    
    atr_15d = calculate_atr(data, window=15)
    
    # Final factor: volatility-normalized signal
    alpha_factor = adjusted_signal / atr_15d.replace(0, np.nan)
    
    # Handle any remaining NaN values
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
