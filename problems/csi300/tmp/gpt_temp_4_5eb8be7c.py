import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Momentum Divergence with Volume Confirmation factor
    Combines short-term and medium-term price momentum with volume trends
    and volume-price correlation for signal confirmation.
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Minimum required data points for calculations
    min_periods = max(20, 10)  # 20 for medium-term trend
    
    for i in range(len(data)):
        if i < min_periods:
            factor_values.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # Price Momentum Component
        # Recent Price Trend (t-5 to t-1)
        if i >= 5:
            recent_prices = current_data['close'].iloc[i-5:i].values
            recent_time = np.arange(len(recent_prices))
            recent_slope = linregress(recent_time, recent_prices).slope if len(recent_prices) > 1 else 0
        else:
            recent_slope = 0
            
        # Medium-Term Price Trend (t-20 to t-1)
        if i >= 20:
            medium_prices = current_data['close'].iloc[i-20:i].values
            medium_time = np.arange(len(medium_prices))
            medium_slope = linregress(medium_time, medium_prices).slope if len(medium_prices) > 1 else 0
        else:
            medium_slope = 0
            
        # Volume Momentum Component
        # Recent Volume Trend (t-5 to t-1)
        if i >= 5:
            recent_volume = current_data['volume'].iloc[i-5:i].values
            recent_vol_time = np.arange(len(recent_volume))
            recent_vol_slope = linregress(recent_vol_time, recent_volume).slope if len(recent_volume) > 1 else 0
        else:
            recent_vol_slope = 0
            
        # Volume-Price Correlation (t-10 to t-1)
        if i >= 10:
            # Calculate returns for correlation
            price_window = current_data['close'].iloc[i-10:i]
            volume_window = current_data['volume'].iloc[i-10:i]
            
            # Calculate price returns
            price_returns = price_window.pct_change().dropna()
            volume_values = volume_window.iloc[1:]  # Align with returns
            
            if len(price_returns) > 1 and len(volume_values) > 1:
                vol_price_corr = np.corrcoef(price_returns.values, volume_values.values)[0, 1]
                vol_price_corr = 0 if np.isnan(vol_price_corr) else vol_price_corr
            else:
                vol_price_corr = 0
        else:
            vol_price_corr = 0
            
        # Momentum Divergence Signal
        # Compare Short vs Medium Price Trends
        trend_difference = recent_slope - medium_slope
        
        # Volume Confirmation
        # Check volume trend alignment and correlation strength
        volume_confirmation = 0
        if abs(recent_vol_slope) > 0 and abs(vol_price_corr) > 0.1:
            # Positive confirmation if volume trend aligns with price momentum
            if (trend_difference > 0 and recent_vol_slope > 0) or (trend_difference < 0 and recent_vol_slope < 0):
                volume_confirmation = 1
            # Negative confirmation if volume contradicts price momentum
            elif (trend_difference > 0 and recent_vol_slope < 0) or (trend_difference < 0 and recent_vol_slope > 0):
                volume_confirmation = -1
        
        # Combine signals
        momentum_signal = trend_difference * (1 + 0.5 * volume_confirmation)
        
        # Apply correlation strength weighting
        final_signal = momentum_signal * (1 + abs(vol_price_corr))
        
        factor_values.iloc[i] = final_signal
    
    return factor_values
