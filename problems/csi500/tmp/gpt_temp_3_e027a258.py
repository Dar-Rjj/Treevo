import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Fractal Dimension (PFD) calculation
    def calculate_pfd(price_data, high_data, low_data, window=5):
        path_length = price_data.diff().abs().rolling(window=window, min_periods=window).sum()
        price_range = (high_data - low_data).rolling(window=3, min_periods=3).sum()
        pfd = 1 + np.log(path_length) / np.log(price_range)
        return pfd
    
    # Volume Fractal Dimension (VFD) calculation
    def calculate_vfd(volume_data, window=5):
        path_length = volume_data.diff().abs().rolling(window=window, min_periods=window).sum()
        
        # Calculate volume range using rolling window
        vol_range = pd.Series(index=volume_data.index, dtype=float)
        for i in range(len(volume_data)):
            if i >= 2:
                window_data = volume_data.iloc[max(0, i-2):i+1]
                vol_range.iloc[i] = window_data.max() - window_data.min()
            else:
                vol_range.iloc[i] = np.nan
        
        vfd = 1 + np.log(path_length) / np.log(vol_range)
        return vfd
    
    # Calculate PFD for different timeframes
    pfd_3d = calculate_pfd(df['close'], df['high'], df['low'], window=3)
    pfd_5d = calculate_pfd(df['close'], df['high'], df['low'], window=5)
    
    # Calculate VFD for different timeframes
    vfd_3d = calculate_vfd(df['volume'], window=3)
    vfd_5d = calculate_vfd(df['volume'], window=5)
    
    # Calculate ratios
    pfd_ratio = pfd_3d / pfd_5d
    vfd_ratio = vfd_3d / vfd_5d
    
    # Calculate trends (3-day rolling mean of differences)
    pfd_trend = pfd_3d.diff().rolling(window=3, min_periods=1).mean()
    vfd_trend = vfd_3d.diff().rolling(window=3, min_periods=1).mean()
    
    # Final factor calculation
    factor = (pfd_ratio - vfd_ratio) * np.sign(pfd_trend - vfd_trend)
    
    return factor
