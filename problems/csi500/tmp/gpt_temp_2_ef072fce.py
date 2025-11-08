import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily range (volatility proxy)
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate ATR (Average True Range) for volatility regime classification
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=14, min_periods=1).mean()
    
    # Volume-to-volatility ratio
    df['volume_volatility_ratio'] = df['volume'] / (df['daily_range'] + 1e-8)
    
    # Volume breakout detection (2x 20-day moving average)
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_breakout'] = (df['volume'] > 2 * df['volume_ma_20']).astype(int)
    
    # Volatility regime classification
    df['atr_ma_60'] = df['atr'].rolling(window=60, min_periods=1).mean()
    df['high_vol_regime'] = (df['atr'] > 1.5 * df['atr_ma_60']).astype(int)
    df['low_vol_regime'] = (df['atr'] < 0.7 * df['atr_ma_60']).astype(int)
    
    # Calculate volume anomaly factor
    # In high volatility: emphasize volume-to-volatility ratio
    # In low volatility: emphasize volume breakouts
    # Normal regime: combine both signals
    df['volume_anomaly_factor'] = (
        df['high_vol_regime'] * df['volume_volatility_ratio'] +
        df['low_vol_regime'] * df['volume_breakout'] * 100 +
        ((1 - df['high_vol_regime']) & (1 - df['low_vol_regime'])) * 
        (df['volume_volatility_ratio'] + df['volume_breakout'] * 50)
    )
    
    # Normalize the factor
    factor = df['volume_anomaly_factor'].rolling(window=252, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    return factor
