import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Pressure Intensity
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    
    selling_pressure = (df['high'] - df['close']) / high_low_range
    buying_pressure = (df['close'] - df['low']) / high_low_range
    net_pressure = buying_pressure - selling_pressure
    
    # Volume Pressure Confirmation
    vol_median_3 = df['volume'].rolling(window=3, min_periods=1).median()
    vol_median_10 = df['volume'].rolling(window=10, min_periods=1).median()
    
    short_vol_intensity = df['volume'] / vol_median_3
    medium_vol_intensity = df['volume'] / vol_median_10
    volume_acceleration = short_vol_intensity / medium_vol_intensity
    
    # Gap Momentum
    close_shifted = df['close'].shift(1)
    overnight_gap = (df['open'] - close_shifted) / close_shifted
    intraday_momentum = (df['close'] - df['open']) / df['open']
    gap_persistence = overnight_gap * intraday_momentum
    
    # Composite Factor
    composite = net_pressure * volume_acceleration * gap_persistence
    
    # Rolling rank for cross-sectional comparison
    factor = composite.rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    return factor
