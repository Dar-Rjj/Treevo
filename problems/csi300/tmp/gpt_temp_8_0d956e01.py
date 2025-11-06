import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Convergence Factor
    Combines short-term (5-day), medium-term (10-day), and long-term (20-day) 
    price-volume signals with volatility adjustment
    """
    
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate components for each timeframe
    for i in range(max(20, len(df))):
        if i < 20:
            continue
            
        # Short-term component (5-day window)
        if i >= 5:
            short_price_mean = close.iloc[i-5:i].mean()
            short_volume_mean = volume.iloc[i-5:i].mean()
            short_price_dev = (close.iloc[i] - short_price_mean) / short_price_mean
            short_volume_dev = (volume.iloc[i] - short_volume_mean) / short_volume_mean
            short_price_vol = close.iloc[i-5:i].std()
            short_volume_vol = volume.iloc[i-5:i].std()
            short_vol_product = short_price_vol * short_volume_vol
            short_adjusted = (short_price_dev * short_volume_dev) / short_vol_product if short_vol_product != 0 else 0
            short_normalized = np.tanh(short_adjusted)
        else:
            short_normalized = 0
        
        # Medium-term component (10-day window)
        if i >= 10:
            medium_price_mean = close.iloc[i-10:i].mean()
            medium_volume_mean = volume.iloc[i-10:i].mean()
            medium_price_dev = (close.iloc[i] - medium_price_mean) / medium_price_mean
            medium_volume_dev = (volume.iloc[i] - medium_volume_mean) / medium_volume_mean
            medium_price_vol = close.iloc[i-10:i].std()
            medium_volume_vol = volume.iloc[i-10:i].std()
            medium_vol_product = medium_price_vol * medium_volume_vol
            medium_adjusted = (medium_price_dev * medium_volume_dev) / medium_vol_product if medium_vol_product != 0 else 0
            medium_normalized = np.tanh(medium_adjusted)
        else:
            medium_normalized = 0
        
        # Long-term component (20-day window)
        long_price_mean = close.iloc[i-20:i].mean()
        long_volume_mean = volume.iloc[i-20:i].mean()
        long_price_dev = (close.iloc[i] - long_price_mean) / long_price_mean
        long_volume_dev = (volume.iloc[i] - long_volume_mean) / long_volume_mean
        long_price_vol = close.iloc[i-20:i].std()
        long_volume_vol = volume.iloc[i-20:i].std()
        long_vol_product = long_price_vol * long_volume_vol
        long_adjusted = (long_price_dev * long_volume_dev) / long_vol_product if long_vol_product != 0 else 0
        long_normalized = np.tanh(long_adjusted)
        
        # Final factor - Multi-Timeframe Convergence
        result.iloc[i] = short_normalized * medium_normalized * long_normalized
    
    return result
