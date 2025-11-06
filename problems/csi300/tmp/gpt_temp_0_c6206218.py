import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime Transition Alpha with Volume Asymmetry
    """
    # Make copy to avoid modifying original dataframe
    data = df.copy()
    
    # Multi-Timeframe Volatility Calculation
    def true_range(high, low, close_prev):
        return np.maximum(high - low, 
                         np.maximum(np.abs(high - close_prev), 
                                   np.abs(low - close_prev)))
    
    # Calculate True Range for different windows
    data['prev_close'] = data['close'].shift(1)
    data['TR'] = true_range(data['high'], data['low'], data['prev_close'])
    
    # Volatility windows
    windows = [3, 5, 10]
    for window in windows:
        data[f'vol_{window}d'] = data['TR'].rolling(window=window).mean()
    
    # Volatility ratios
    data['vol_ratio_5_3'] = data['vol_5d'] / data['vol_3d']
    data['vol_ratio_10_5'] = data['vol_10d'] / data['vol_5d']
    
    # Regime Change Detection
    def detect_regime_shifts(vol_series, threshold=1.5, lookback=5):
        shifts = pd.Series(0, index=vol_series.index)
        for i in range(lookback, len(vol_series)):
            current_vol = vol_series.iloc[i]
            past_vol = vol_series.iloc[i-lookback:i].mean()
            if current_vol > past_vol * threshold:
                shifts.iloc[i] = 1  # High volatility regime
            elif current_vol < past_vol / threshold:
                shifts.iloc[i] = -1  # Low volatility regime
        return shifts
    
    # Detect regime shifts using 5-day volatility
    data['regime_shift'] = detect_regime_shifts(data['vol_5d'])
    
    # Create transition periods (3 days before/after shift)
    data['transition_flag'] = 0
    for i in range(len(data)):
        if data['regime_shift'].iloc[i] != 0:
            # Mark transition period
            start_idx = max(0, i-3)
            end_idx = min(len(data)-1, i+3)
            data.iloc[start_idx:end_idx+1, data.columns.get_loc('transition_flag')] = 1
    
    # Price-Volume Asymmetry Calculation
    def directional_volume_flow(close, volume, lookback=5):
        price_change = close.diff()
        up_volume = volume.where(price_change > 0, 0)
        down_volume = volume.where(price_change < 0, 0)
        
        # Rolling sums
        up_vol_roll = up_volume.rolling(window=lookback).sum()
        down_vol_roll = down_volume.rolling(window=lookback).sum()
        
        # Volume flow ratio
        vol_flow = (up_vol_roll - down_vol_roll) / (up_vol_roll + down_vol_roll + 1e-8)
        return vol_flow
    
    data['vol_flow'] = directional_volume_flow(data['close'], data['volume'])
    
    # Volume skewness during transitions
    def transition_volume_skewness(volume, transition_flag, window=5):
        skewness = pd.Series(0.0, index=volume.index)
        for i in range(window, len(volume)):
            if transition_flag.iloc[i] == 1:
                # Calculate volume skewness in transition window
                vol_window = volume.iloc[i-window:i+1]
                if len(vol_window) > 2:
                    mean_vol = vol_window.mean()
                    std_vol = vol_window.std()
                    if std_vol > 0:
                        skewness.iloc[i] = ((vol_window - mean_vol) ** 3).mean() / (std_vol ** 3)
        return skewness
    
    data['vol_skew'] = transition_volume_skewness(data['volume'], data['transition_flag'])
    
    # Volume concentration measure
    def volume_concentration(close, volume, window=3):
        returns = close.pct_change()
        directional_volume = volume * np.sign(returns)
        concentration = directional_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()
        return concentration.fillna(0)
    
    data['vol_concentration'] = volume_concentration(data['close'], data['volume'])
    
    # Generate Transition Alpha
    def time_decay_weight(distance_from_shift, max_distance=6):
        """Weight signals by proximity to transition point"""
        if distance_from_shift == 0:
            return 1.0
        return max(0, 1 - abs(distance_from_shift) / max_distance)
    
    # Calculate distance from nearest regime shift
    data['distance_to_shift'] = 0
    last_shift_idx = -10
    for i in range(len(data)):
        if data['regime_shift'].iloc[i] != 0:
            last_shift_idx = i
        if last_shift_idx >= 0:
            data.iloc[i, data.columns.get_loc('distance_to_shift')] = i - last_shift_idx
    
    # Apply time decay weighting
    data['time_weight'] = data['distance_to_shift'].apply(time_decay_weight)
    
    # Combine signals for final alpha
    # Volume asymmetry strength (absolute value for magnitude)
    vol_asymmetry = np.abs(data['vol_flow'] * data['vol_skew'] * data['vol_concentration'])
    
    # Adjust for transition direction
    regime_direction = np.sign(data['regime_shift'])
    
    # Final alpha: transition flag * volume asymmetry * regime direction * time weight
    alpha = (data['transition_flag'] * vol_asymmetry * regime_direction * data['time_weight'])
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha
