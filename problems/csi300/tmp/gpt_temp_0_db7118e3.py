import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Pressure
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    
    buying_pressure = (df['close'] - df['low']) / high_low_range
    selling_pressure = (df['high'] - df['close']) / high_low_range
    net_pressure = buying_pressure - selling_pressure
    
    # Volume Confirmation
    vol_median_3 = df['volume'].rolling(window=3, min_periods=1).median()
    vol_median_10 = df['volume'].rolling(window=10, min_periods=1).median()
    
    short_term_intensity = df['volume'] / vol_median_3
    medium_term_intensity = df['volume'] / vol_median_10
    volume_acceleration = short_term_intensity / medium_term_intensity
    
    # Gap Momentum
    close_shifted = df['close'].shift(1)
    overnight_gap = (df['open'] - close_shifted) / close_shifted
    intraday_momentum = (df['close'] - df['open']) / df['open']
    gap_persistence = overnight_gap * intraday_momentum
    
    # Composite Signal
    composite = net_pressure * volume_acceleration * gap_persistence
    
    # Cross-sectional rank
    factor = composite.rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 1 else np.nan
    )
    
    return factor
