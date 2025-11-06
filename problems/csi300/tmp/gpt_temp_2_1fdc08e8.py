import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Overnight Gap Ratio
    df['gap_ratio'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate 10-day Average True Range
    df['atr_10'] = df['true_range'].rolling(window=10, min_periods=10).mean()
    
    # Calculate Volatility Z-score
    df['atr_mean_20'] = df['atr_10'].rolling(window=20, min_periods=20).mean()
    df['atr_std_20'] = df['atr_10'].rolling(window=20, min_periods=20).std()
    df['vol_z_score'] = (df['atr_10'] - df['atr_mean_20']) / df['atr_std_20']
    
    # Identify Extreme Gaps using deciles
    df['gap_decile'] = df['gap_ratio'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else np.nan,
        raw=False
    )
    
    # Generate Reversion Signal
    df['signal'] = -np.sign(df['gap_ratio']) * np.abs(df['gap_ratio'])
    
    # Apply volatility filter and volume confirmation
    df['volume_mean_5'] = df['volume'].rolling(window=5, min_periods=5).mean()
    
    # Combine all conditions
    extreme_gap = (df['gap_decile'] == 0) | (df['gap_decile'] == 9)
    high_vol = np.abs(df['vol_z_score']) > 1
    high_volume = df['volume'] > df['volume_mean_5']
    
    # Final factor calculation
    factor = df['signal'] * extreme_gap * high_vol * high_volume
    
    return factor
