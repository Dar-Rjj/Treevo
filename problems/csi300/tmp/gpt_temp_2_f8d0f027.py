import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Multi-Timeframe Momentum Divergence Factor
    Combines short-term (5-day) and medium-term (10-day) momentum signals
    with volatility adjustments and divergence analysis
    """
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate components for each day
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of data
            result.iloc[i] = 0
            continue
            
        # Short-term components (5-day)
        # Price momentum
        short_price_return = (close.iloc[i] - close.iloc[i-4]) / close.iloc[i-4]
        short_price_vol = close.iloc[i-4:i+1].std()
        short_price_momentum = short_price_return / short_price_vol if short_price_vol != 0 else 0
        
        # Volume momentum
        short_volume_change = (volume.iloc[i] - volume.iloc[i-4]) / volume.iloc[i-4] if volume.iloc[i-4] != 0 else 0
        short_volume_vol = volume.iloc[i-4:i+1].std()
        short_volume_momentum = short_volume_change / short_volume_vol if short_volume_vol != 0 else 0
        
        # Medium-term components (10-day)
        # Price momentum
        medium_price_return = (close.iloc[i] - close.iloc[i-9]) / close.iloc[i-9]
        medium_price_vol = close.iloc[i-9:i+1].std()
        medium_price_momentum = medium_price_return / medium_price_vol if medium_price_vol != 0 else 0
        
        # Volume momentum
        medium_volume_change = (volume.iloc[i] - volume.iloc[i-9]) / volume.iloc[i-9] if volume.iloc[i-9] != 0 else 0
        medium_volume_vol = volume.iloc[i-9:i+1].std()
        medium_volume_momentum = medium_volume_change / medium_volume_vol if medium_volume_vol != 0 else 0
        
        # Momentum divergence calculation
        price_divergence = short_price_momentum - medium_price_momentum
        volume_divergence = short_volume_momentum - medium_volume_momentum
        
        # Final factor construction
        raw_signal = price_divergence * volume_divergence
        
        # Volatility scaling
        recent_vol = close.iloc[i-4:i+1].std()
        scaled_signal = raw_signal / recent_vol if recent_vol != 0 else 0
        
        # Bound output for robustness
        bounded_signal = np.tanh(scaled_signal) * recent_vol
        
        result.iloc[i] = bounded_signal
    
    return result
