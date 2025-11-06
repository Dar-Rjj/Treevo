import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Momentum Alignment Factor
    Combines short-term (5-day) and medium-term (10-day) price and volume momentum
    with volatility normalization and alignment through multiplication.
    """
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate components for each time period
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days for medium-term calculation
            result.iloc[i] = 0
            continue
            
        # Short-term components (5-day window)
        # Price momentum with volatility adjustment
        if i >= 5:
            short_price_return = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5]
            short_price_vol = close.iloc[i-5:i].std()
            short_price_momentum = short_price_return / short_price_vol if short_price_vol != 0 else 0
            
            # Volume momentum with volatility adjustment
            short_volume_change = (volume.iloc[i] - volume.iloc[i-5]) / volume.iloc[i-5]
            short_volume_vol = volume.iloc[i-5:i].std()
            short_volume_momentum = short_volume_change / short_volume_vol if short_volume_vol != 0 else 0
            
            # Short-term signal (aligned components)
            short_signal = short_price_momentum * short_volume_momentum
        else:
            short_signal = 0
        
        # Medium-term components (10-day window)
        # Price momentum with volatility adjustment
        medium_price_return = (close.iloc[i] - close.iloc[i-10]) / close.iloc[i-10]
        medium_price_vol = close.iloc[i-10:i].std()
        medium_price_momentum = medium_price_return / medium_price_vol if medium_price_vol != 0 else 0
        
        # Volume momentum with volatility adjustment
        medium_volume_change = (volume.iloc[i] - volume.iloc[i-10]) / volume.iloc[i-10]
        medium_volume_vol = volume.iloc[i-10:i].std()
        medium_volume_momentum = medium_volume_change / medium_volume_vol if medium_volume_vol != 0 else 0
        
        # Medium-term signal (aligned components)
        medium_signal = medium_price_momentum * medium_volume_momentum
        
        # Combine timeframes
        combined_signal = short_signal * medium_signal
        
        # Apply stability enhancement
        result.iloc[i] = np.tanh(combined_signal)
    
    return result
