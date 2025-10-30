import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Momentum Reversal with Volume Confirmation factor
    Identifies potential reversal opportunities by combining price momentum extremes with volume trend confirmation
    """
    df = data.copy()
    
    # Calculate 5-day price momentum (t-5 to t-1)
    df['momentum_5d'] = (df['close'].shift(1) - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate 10-day volume slope using linear regression
    def calc_volume_slope(volume_series):
        if len(volume_series) < 10 or volume_series.isna().any():
            return np.nan
        x = np.arange(10)
        y = volume_series.values
        slope, _, _, _, _ = linregress(x, y)
        return slope
    
    # Create volume series for slope calculation
    volume_data = pd.DataFrame(index=df.index)
    for i in range(10):
        volume_data[f'volume_lag_{i}'] = df['volume'].shift(i+1)
    
    df['volume_slope'] = volume_data.apply(calc_volume_slope, axis=1)
    
    # Identify momentum extremes (top and bottom 20%)
    df['momentum_rank'] = df['momentum_5d'].rank(pct=True)
    df['is_strong_winner'] = df['momentum_rank'] >= 0.8
    df['is_strong_loser'] = df['momentum_rank'] <= 0.2
    
    # Apply volume confirmation logic
    factor_values = []
    
    for idx, row in df.iterrows():
        momentum = row['momentum_5d']
        volume_slope = row['volume_slope']
        is_winner = row['is_strong_winner']
        is_loser = row['is_strong_loser']
        
        if pd.isna(momentum) or pd.isna(volume_slope):
            factor_values.append(np.nan)
            continue
        
        # Strong winners with declining volume (negative slope)
        if is_winner and volume_slope < 0:
            # Negative momentum (reversal signal) multiplied by negative volume slope
            signal = (-momentum) * volume_slope
            # Scale by absolute momentum to emphasize stronger reversals
            factor_value = signal * abs(momentum)
        
        # Strong losers with increasing volume (positive slope)
        elif is_loser and volume_slope > 0:
            # Positive momentum (reversal signal) multiplied by positive volume slope
            signal = (-momentum) * volume_slope  # -momentum gives positive for losers
            # Scale by absolute momentum to emphasize stronger reversals
            factor_value = signal * abs(momentum)
        
        else:
            factor_value = 0.0
        
        factor_values.append(factor_value)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=df.index, name='momentum_reversal_volume')
    
    return factor_series
