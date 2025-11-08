import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Regime Classification
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['daily_range'] = df['high'] - df['low']
    df['range_vol_5'] = df['daily_range'].rolling(window=5).std()
    df['volatility_regime'] = (df['atr_5'] > df['atr_5'].rolling(window=20).mean()).astype(int)
    
    # Momentum Acceleration Core
    df['mom_3'] = df['close'] / df['close'].shift(3) - 1
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['mom_accel_3'] = df['mom_3'] - df['mom_3'].shift(3)
    df['mom_accel_10'] = df['mom_10'] - df['mom_10'].shift(3)
    
    # Volume Confirmation Signals
    df['volume_slope'] = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    df['price_impact'] = df['true_range'] / (df['volume'] + 1e-8)
    
    # Regime-Adaptive Signal Blending
    # High-volatility regime: volatility-scaled momentum
    high_vol_factor = (df['mom_10'] / (df['atr_5'] + 1e-8)) * df['volume_slope']
    
    # Low-volatility regime: price acceleration with volume confirmation
    low_vol_factor = (df['mom_accel_3'] + df['mom_accel_10']) * (1 + df['volume_slope']) * (1 - df['price_impact'])
    
    # Combine regimes
    factor = df['volatility_regime'] * high_vol_factor + (1 - df['volatility_regime']) * low_vol_factor
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'atr_5', 
                   'daily_range', 'range_vol_5', 'volatility_regime', 'mom_3', 
                   'mom_10', 'mom_accel_3', 'mom_accel_10', 'volume_slope', 'price_impact']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return factor
