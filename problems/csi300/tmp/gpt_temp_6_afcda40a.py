import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Non-Overlapping Trend Consistency Alpha
    
    This factor captures the consistency of price, volume, and amount trends
    across non-overlapping periods, validated by cross-signal alignment and
    momentum strength.
    """
    # Extract required columns
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate slopes and directions for each time period
    for i in range(5, len(df)):
        # Price Trend Analysis
        recent_price_slope = close.iloc[i-2:i+1].diff().mean()
        prior_price_slope = close.iloc[i-5:i-2].diff().mean()
        recent_price_dir = np.sign(recent_price_slope)
        prior_price_dir = np.sign(prior_price_slope)
        
        # Volume Trend Analysis
        recent_volume_slope = volume.iloc[i-2:i+1].diff().mean()
        prior_volume_slope = volume.iloc[i-5:i-2].diff().mean()
        recent_volume_dir = np.sign(recent_volume_slope)
        prior_volume_dir = np.sign(prior_volume_slope)
        
        # Amount Trend Analysis
        recent_amount_slope = amount.iloc[i-2:i+1].diff().mean()
        prior_amount_slope = amount.iloc[i-5:i-2].diff().mean()
        recent_amount_dir = np.sign(recent_amount_slope)
        prior_amount_dir = np.sign(prior_amount_slope)
        
        # Trend Consistency Signals
        price_consistency = 1 if recent_price_dir == prior_price_dir else -1
        volume_consistency = 1 if recent_volume_dir == prior_volume_dir else -1
        amount_consistency = 1 if recent_amount_dir == prior_amount_dir else -1
        
        # Cross-Signal Validation
        price_volume_alignment = 1 if recent_price_dir == recent_volume_dir else -1
        price_amount_alignment = 1 if recent_price_dir == recent_amount_dir else -1
        volume_amount_alignment = 1 if recent_volume_dir == recent_amount_dir else -1
        
        # Trend Strength Measurement
        price_strength = abs(recent_price_slope)
        volume_strength = abs(recent_volume_slope)
        amount_strength = abs(recent_amount_slope)
        
        # Composite Alpha Generation
        base_consistency_score = price_consistency * volume_consistency * amount_consistency
        cross_validation_score = price_volume_alignment * price_amount_alignment * volume_amount_alignment
        strength_multiplier = price_strength * volume_strength * amount_strength
        
        # Final Alpha
        alpha.iloc[i] = base_consistency_score * cross_validation_score * strength_multiplier
    
    # Handle NaN values at the beginning
    alpha = alpha.fillna(0)
    
    return alpha
