import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Momentum Alignment Factor
    Combines short-term (5-day) and medium-term (10-day) momentum signals
    normalized by their respective volatilities and stabilized with tanh
    """
    # Extract necessary columns
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate components for each date
    for i in range(10, len(df)):
        current_date = df.index[i]
        
        # Short-term components (5-day window)
        if i >= 5:
            # Price momentum components
            price_return_5d = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5]
            price_vol_5d = close.iloc[i-5:i].std()
            
            # Volume momentum components
            volume_return_5d = (volume.iloc[i] - volume.iloc[i-5]) / volume.iloc[i-5]
            volume_vol_5d = volume.iloc[i-5:i].std()
            
            # Normalize and combine short-term signals
            if price_vol_5d > 0 and volume_vol_5d > 0:
                norm_price_momentum_5d = price_return_5d / price_vol_5d
                norm_volume_momentum_5d = volume_return_5d / volume_vol_5d
                short_term_signal = norm_price_momentum_5d * norm_volume_momentum_5d
            else:
                short_term_signal = 0
        else:
            short_term_signal = 0
        
        # Medium-term components (10-day window)
        # Price momentum components
        price_return_10d = (close.iloc[i] - close.iloc[i-10]) / close.iloc[i-10]
        price_vol_10d = close.iloc[i-10:i].std()
        
        # Volume momentum components
        volume_return_10d = (volume.iloc[i] - volume.iloc[i-10]) / volume.iloc[i-10]
        volume_vol_10d = volume.iloc[i-10:i].std()
        
        # Normalize and combine medium-term signals
        if price_vol_10d > 0 and volume_vol_10d > 0:
            norm_price_momentum_10d = price_return_10d / price_vol_10d
            norm_volume_momentum_10d = volume_return_10d / volume_vol_10d
            medium_term_signal = norm_price_momentum_10d * norm_volume_momentum_10d
        else:
            medium_term_signal = 0
        
        # Combine and stabilize signals
        combined_signal = short_term_signal * medium_term_signal
        factor.iloc[i] = np.tanh(combined_signal)
    
    return factor
