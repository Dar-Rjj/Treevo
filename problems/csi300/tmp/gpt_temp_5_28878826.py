import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute Range Persistence Score
    tr_std_5 = true_range.rolling(window=5).std()
    tr_std_20 = true_range.rolling(window=20).std()
    range_persistence = tr_std_5 / tr_std_20
    
    # Measure Volume Acceleration
    vol_avg_5 = df['volume'].rolling(window=5).mean()
    vol_avg_20 = df['volume'].rolling(window=20).mean()
    volume_acceleration = vol_avg_5 / vol_avg_20
    
    # Combine Signals
    combined_signal = range_persistence * volume_acceleration
    
    # Apply rolling rank
    factor = combined_signal.rolling(window=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], 
        raw=False
    )
    
    return factor
