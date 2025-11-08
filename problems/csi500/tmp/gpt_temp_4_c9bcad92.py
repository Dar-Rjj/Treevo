import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required intermediate series
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Calculate daily price and volume direction indicators
    price_up = (close > close.shift(1)).astype(int)
    volume_up = (volume > volume.shift(1)).astype(int)
    
    # Initialize rolling windows for volatility calculations
    window = 20  # Using 20-day window for volatility calculations
    
    for i in range(window, len(df)):
        # Volatility Asymmetry Analysis
        window_data = df.iloc[i-window:i]
        
        # Upward volatility (only up days)
        up_days = window_data[window_data['close'] > window_data['open']]
        if len(up_days) > 0:
            upward_vol = np.sqrt(np.sum((up_days['high'] - up_days['open'])**2) / len(up_days))
        else:
            upward_vol = 0.001  # Small value to avoid division by zero
            
        # Downward volatility (only down days)
        down_days = window_data[window_data['close'] < window_data['open']]
        if len(down_days) > 0:
            downward_vol = np.sqrt(np.sum((down_days['open'] - down_days['low'])**2) / len(down_days))
        else:
            downward_vol = 0.001  # Small value to avoid division by zero
        
        # Momentum Entropy Measurement
        # 3-day price direction entropy
        if i >= 3:
            price_3d_window = price_up.iloc[i-3:i]
            p_up_3d = np.sum(price_3d_window) / 3
            p_down_3d = 1 - p_up_3d
            
            if p_up_3d > 0 and p_down_3d > 0:
                price_entropy_3d = -(p_up_3d * np.log(p_up_3d) + p_down_3d * np.log(p_down_3d))
            else:
                price_entropy_3d = 0
        else:
            price_entropy_3d = 0
            
        # 5-day price direction entropy
        if i >= 5:
            price_5d_window = price_up.iloc[i-5:i]
            p_up_5d = np.sum(price_5d_window) / 5
            p_down_5d = 1 - p_up_5d
            
            if p_up_5d > 0 and p_down_5d > 0:
                price_entropy_5d = -(p_up_5d * np.log(p_up_5d) + p_down_5d * np.log(p_down_5d))
            else:
                price_entropy_5d = 0
        else:
            price_entropy_5d = 0
            
        # 3-day volume direction entropy
        if i >= 3:
            volume_3d_window = volume_up.iloc[i-3:i]
            v_up_3d = np.sum(volume_3d_window) / 3
            v_down_3d = 1 - v_up_3d
            
            if v_up_3d > 0 and v_down_3d > 0:
                volume_entropy_3d = -(v_up_3d * np.log(v_up_3d) + v_down_3d * np.log(v_down_3d))
            else:
                volume_entropy_3d = 0
        else:
            volume_entropy_3d = 0
            
        # 5-day volume direction entropy
        if i >= 5:
            volume_5d_window = volume_up.iloc[i-5:i]
            v_up_5d = np.sum(volume_5d_window) / 5
            v_down_5d = 1 - v_up_5d
            
            if v_up_5d > 0 and v_down_5d > 0:
                volume_entropy_5d = -(v_up_5d * np.log(v_up_5d) + v_down_5d * np.log(v_down_5d))
            else:
                volume_entropy_5d = 0
        else:
            volume_entropy_5d = 0
        
        # Asymmetric Volatility Ratio
        volatility_skew = upward_vol / (downward_vol + 1e-8)
        volatility_efficiency = (high.iloc[i] - low.iloc[i]) / (upward_vol + downward_vol + 1e-8)
        
        # Entropy Convergence Patterns
        price_volume_divergence = price_entropy_3d - volume_entropy_3d
        entropy_gradient = (price_entropy_5d - price_entropy_3d) * (volume_entropy_5d - volume_entropy_3d)
        
        # Asymmetric Momentum Integration
        core_factor = volatility_skew * price_volume_divergence
        
        # Volatility-Regime Enhancement
        if volatility_efficiency > 1.0:  # High efficiency regime
            factor_value = core_factor * volatility_efficiency * abs(entropy_gradient)
        else:  # Low efficiency regime
            factor_value = core_factor * (1 + abs(price_volume_divergence)) / (volatility_efficiency + 1e-8)
        
        result.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
