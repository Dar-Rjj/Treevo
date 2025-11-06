import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Divergence with Trend Acceleration alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Minimum required window for calculations
    min_window = 10
    
    for i in range(min_window, len(data)):
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Compute Price Trend Acceleration
        # Short-term price trend (5-day)
        if i >= 4:
            short_term_prices = current_data['close'].iloc[i-4:i+1].values
            short_term_idx = np.arange(len(short_term_prices))
            short_slope, _, _, _, _ = linregress(short_term_idx, short_term_prices)
        else:
            short_slope = 0
        
        # Medium-term price trend (10-day)
        if i >= 9:
            medium_term_prices = current_data['close'].iloc[i-9:i+1].values
            medium_term_idx = np.arange(len(medium_term_prices))
            medium_slope, _, _, _, _ = linregress(medium_term_idx, medium_term_prices)
        else:
            medium_slope = 0
        
        # Trend acceleration
        trend_acceleration = short_slope - medium_slope
        
        # 2. Compute Volume-Price Divergence
        volume_divergence = 0
        if i >= 4:
            # Volume trend (5-day)
            volume_data = current_data['volume'].iloc[i-4:i+1].values
            volume_idx = np.arange(len(volume_data))
            volume_slope, _, _, _, _ = linregress(volume_idx, volume_data)
            
            # Price-Volume correlation (5-day)
            price_returns = np.diff(current_data['close'].iloc[i-4:i+1].values)
            volume_changes = np.diff(volume_data)
            
            if len(price_returns) >= 2 and len(volume_changes) >= 2:
                correlation = np.corrcoef(price_returns, volume_changes)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            # Identify divergence
            price_trend_sign = np.sign(short_slope)
            volume_trend_sign = np.sign(volume_slope)
            
            # Divergence conditions
            if price_trend_sign != volume_trend_sign:
                # Opposite trends
                divergence_magnitude = abs(short_slope * volume_slope)
                volume_divergence = divergence_magnitude * price_trend_sign
            elif correlation < 0 and abs(short_slope) > 0 and abs(volume_slope) > 0:
                # Negative correlation with trending series
                divergence_magnitude = abs(short_slope * volume_slope * correlation)
                volume_divergence = divergence_magnitude * price_trend_sign
            else:
                volume_divergence = 0
        
        # 3. Compute Intraday Strength Signal
        current_row = current_data.iloc[-1]
        high_low_range = current_row['high'] - current_row['low']
        
        if high_low_range > 0:
            # Close-to-Open strength
            close_open_strength = (current_row['close'] - current_row['open']) / high_low_range
            
            # High-Low efficiency
            high_low_efficiency = abs(current_row['close'] - current_row['open']) / high_low_range
            
            # Combined intraday signal
            intraday_signal = close_open_strength * high_low_efficiency
        else:
            intraday_signal = 0
        
        # 4. Synthesize Composite Alpha
        # Base signal
        base_signal = trend_acceleration * volume_divergence * intraday_signal
        
        # Directional adjustment
        if volume_divergence > 0 and trend_acceleration > 0:
            # Bullish scenario - enhance signal
            directional_adjustment = 1.5
        elif volume_divergence < 0 and trend_acceleration < 0:
            # Bearish scenario - enhance signal
            directional_adjustment = 1.5
        else:
            # Reduce signal magnitude for conflicting signals
            directional_adjustment = 0.7
        
        final_alpha = base_signal * directional_adjustment
        
        # Store result
        alpha.iloc[i] = final_alpha
    
    # Fill initial values with 0
    alpha = alpha.fillna(0)
    
    return alpha
