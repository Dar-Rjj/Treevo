import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Weighted Trend Convergence factor
    """
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(60, len(df)):
        current_idx = df.index[i]
        
        # Long-Term Trend Component (60-day)
        long_term_close = close.iloc[i-60:i]
        long_term_volume = volume.iloc[i-60:i]
        
        # Price Trend Strength
        if len(long_term_close) >= 2:
            price_slope, _, _, _, _ = linregress(range(len(long_term_close)), long_term_close.values)
            price_trend_magnitude = abs(price_slope)
            price_trend_direction = np.sign(price_slope)
        else:
            price_trend_magnitude = 0
            price_trend_direction = 0
        
        # Volume Trend Strength
        if len(long_term_volume) >= 2:
            volume_slope, _, _, _, _ = linregress(range(len(long_term_volume)), long_term_volume.values)
            volume_trend_magnitude = abs(volume_slope)
            volume_trend_direction = np.sign(volume_slope)
        else:
            volume_trend_magnitude = 0
            volume_trend_direction = 0
        
        # Medium-Term Trend Component (20-day)
        if i >= 20:
            # Price Momentum
            price_roc = (close.iloc[i] - close.iloc[i-20]) / close.iloc[i-20]
            price_momentum_direction = np.sign(price_roc)
            
            if i >= 20:
                price_acceleration = (close.iloc[i] - 2*close.iloc[i-10] + close.iloc[i-20]) / close.iloc[i-20]
            else:
                price_acceleration = 0
            
            # Volume Momentum
            volume_roc = (volume.iloc[i] - volume.iloc[i-20]) / volume.iloc[i-20]
            volume_momentum_direction = np.sign(volume_roc)
            
            if i >= 20:
                volume_acceleration = (volume.iloc[i] - 2*volume.iloc[i-10] + volume.iloc[i-20]) / volume.iloc[i-20]
            else:
                volume_acceleration = 0
        else:
            price_roc = price_acceleration = volume_roc = volume_acceleration = 0
            price_momentum_direction = volume_momentum_direction = 0
        
        # Volatility Framework
        # Price Volatility Structure
        long_term_price_vol = close.iloc[i-60:i].std() if i >= 60 else 1
        medium_term_price_vol = close.iloc[i-20:i].std() if i >= 20 else 1
        short_term_price_vol = close.iloc[i-5:i].std() if i >= 5 else 1
        
        # Volume Volatility Structure
        long_term_volume_vol = volume.iloc[i-60:i].std() if i >= 60 else 1
        medium_term_volume_vol = volume.iloc[i-20:i].std() if i >= 20 else 1
        short_term_volume_vol = volume.iloc[i-5:i].std() if i >= 5 else 1
        
        # Avoid division by zero
        long_term_price_vol = max(long_term_price_vol, 1e-6)
        medium_term_price_vol = max(medium_term_price_vol, 1e-6)
        long_term_volume_vol = max(long_term_volume_vol, 1e-6)
        medium_term_volume_vol = max(medium_term_volume_vol, 1e-6)
        
        # Trend Convergence Scoring
        long_term_alignment = price_trend_direction * volume_trend_direction
        medium_term_alignment = price_momentum_direction * volume_momentum_direction
        
        # Volatility-Weighted Integration
        long_term_trend_strength = (price_trend_magnitude * volume_trend_magnitude) / (long_term_price_vol * long_term_volume_vol)
        medium_term_momentum_strength = (abs(price_acceleration) * abs(volume_acceleration)) / (medium_term_price_vol * medium_term_volume_vol)
        
        directional_consistency = long_term_alignment * medium_term_alignment
        
        # Final factor calculation
        factor_value = directional_consistency * (long_term_trend_strength + medium_term_momentum_strength)
        factor.loc[current_idx] = factor_value
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
