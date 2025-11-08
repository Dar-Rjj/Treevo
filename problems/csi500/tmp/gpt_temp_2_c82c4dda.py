import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price returns for correlation
    data['price_returns'] = data['close'].pct_change()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 19:  # Need at least 20 days for calculations
            result.iloc[i] = 0
            continue
            
        current_idx = i
        window_5 = slice(max(0, i-4), i+1)
        window_10 = slice(max(0, i-9), i+1)
        window_20 = slice(max(0, i-19), i+1)
        
        # Get data for current windows
        close_prices = data['close'].iloc[window_10].values
        volumes = data['volume'].iloc[window_10].values
        days = np.arange(len(close_prices))
        
        # 1. Compute Price Trend Strength
        # 5-day price slope
        if len(close_prices[-5:]) >= 2:
            price_slope_5 = linregress(np.arange(5), close_prices[-5:]).slope
        else:
            price_slope_5 = 0
            
        # 10-day price slope
        if len(close_prices) >= 2:
            price_slope_10 = linregress(days, close_prices).slope
        else:
            price_slope_10 = 0
            
        # Slope ratio for trend acceleration
        if price_slope_10 != 0:
            price_slope_ratio = price_slope_5 / price_slope_10
        else:
            price_slope_ratio = 0
            
        # 2. Calculate Volume Confirmation
        # 5-day volume slope
        if len(volumes[-5:]) >= 2:
            volume_slope_5 = linregress(np.arange(5), volumes[-5:]).slope
        else:
            volume_slope_5 = 0
            
        # 10-day volume slope
        if len(volumes) >= 2:
            volume_slope_10 = linregress(days, volumes).slope
        else:
            volume_slope_10 = 0
            
        # Volume slope ratio
        if volume_slope_10 != 0:
            volume_slope_ratio = volume_slope_5 / volume_slope_10
        else:
            volume_slope_ratio = 0
            
        # Price-Volume Convergence
        price_volume_convergence = price_slope_ratio * volume_slope_ratio
        
        # 3. Detect Regime Switching Patterns
        # Calculate intraday price range percentage
        current_range = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i]
        
        # Calculate 20-day median of daily range percentages
        range_window = data.iloc[window_20]
        daily_ranges = (range_window['high'] - range_window['low']) / range_window['close']
        median_range = daily_ranges.median()
        
        # Regime indicator (1 if high volatility, 0 otherwise)
        regime_indicator = 1 if current_range > median_range else 0
        
        # 4. Calculate Volume-Price Divergence
        # Get data for correlation calculation
        vol_window = data['volume'].iloc[window_10].values
        returns_window = data['price_returns'].iloc[window_10].values
        
        # Remove NaN values for correlation
        valid_mask = ~np.isnan(returns_window)
        vol_valid = vol_window[valid_mask]
        returns_valid = returns_window[valid_mask]
        
        if len(vol_valid) >= 2 and len(returns_valid) >= 2:
            correlation = np.corrcoef(vol_valid, returns_valid)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
            
        # Multiply by sign of price trend
        signed_correlation = correlation * np.sign(price_slope_10)
        
        # Divergence strength
        divergence_strength = signed_correlation * price_slope_ratio
        
        # 5. Generate Adaptive Alpha Factor
        # Combine convergence and regime signals
        regime_adjusted_base = price_volume_convergence * regime_indicator
        
        # Apply divergence weighting
        weighted_signal = regime_adjusted_base * divergence_strength
        
        # Dynamic smoothing with adaptive window
        window_size = 3 if regime_indicator == 1 else 5
        
        # Calculate moving average with selected window
        start_idx = max(0, i - window_size + 1)
        signal_window = result.iloc[start_idx:i+1]
        
        if len(signal_window) > 0:
            smoothed_value = signal_window.mean()
        else:
            smoothed_value = weighted_signal
            
        result.iloc[i] = smoothed_value
    
    return result
