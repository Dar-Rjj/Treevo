import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Component
    # Short-term momentum (5-day)
    data['short_momentum'] = data['close'].pct_change(periods=5)
    
    # Medium-term momentum (20-day)
    data['medium_momentum'] = data['close'].pct_change(periods=20)
    
    # Volume Trend Component
    # Calculate volume trend using linear regression slope (10-day window)
    def volume_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
    
    data['volume_trend'] = data['volume'].rolling(window=10, min_periods=2).apply(
        volume_slope, raw=False
    )
    
    # Volume breakout ratio (current volume vs 20-day average)
    data['volume_avg_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_avg_20']
    
    # Divergence Detection
    # Compare momentum directions
    data['momentum_alignment'] = np.sign(data['short_momentum']) * np.sign(data['medium_momentum'])
    
    # Calculate divergence score
    def calculate_divergence_score(row):
        if pd.isna(row['short_momentum']) or pd.isna(row['medium_momentum']) or pd.isna(row['volume_trend']):
            return np.nan
        
        # Base divergence: difference between short and medium momentum
        momentum_divergence = abs(row['short_momentum'] - row['medium_momentum'])
        
        # Directional conflict penalty
        direction_penalty = 0
        if row['momentum_alignment'] < 0:  # Opposite directions
            direction_penalty = 1.0
        
        # Volume confirmation
        volume_confirmation = 0
        if row['volume_trend'] > 0 and row['volume_ratio'] > 1.2:  # Strong volume confirmation
            volume_confirmation = 1.0
        elif row['volume_trend'] < 0 and row['volume_ratio'] < 0.8:  # Weak volume confirmation
            volume_confirmation = -1.0
        
        # Combine components
        divergence_score = (
            momentum_divergence * 10 +  # Scale divergence
            direction_penalty * 2 +     # Penalty for conflicting directions
            volume_confirmation * 3     # Volume confirmation bonus/penalty
        )
        
        return divergence_score
    
    data['divergence_score'] = data.apply(calculate_divergence_score, axis=1)
    
    # Final factor: divergence score with sign based on short-term momentum direction
    data['factor'] = data['divergence_score'] * np.sign(data['short_momentum'])
    
    return data['factor']
