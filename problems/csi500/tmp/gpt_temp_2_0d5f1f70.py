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
    
    # Normalize volatility measures
    df['atr_norm'] = df['atr_5'] / df['close']
    df['range_vol_norm'] = df['range_vol_5'] / df['close']
    
    # Combined volatility regime score
    df['volatility_regime'] = (df['atr_norm'] + df['range_vol_norm']) / 2
    
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
    
    # Normalize volume signals
    df['volume_slope_norm'] = df['volume_slope'] / (df['volume'].rolling(window=5).mean() + 1e-8)
    df['price_impact_norm'] = df['price_impact'] / (df['price_impact'].rolling(window=5).mean() + 1e-8)
    
    # Volume confirmation score
    df['volume_confirmation'] = df['volume_slope_norm'] * (1 - df['price_impact_norm'])
    
    # Regime-Adaptive Signal Blending
    high_vol_threshold = df['volatility_regime'].quantile(0.7)
    low_vol_threshold = df['volatility_regime'].quantile(0.3)
    
    # High-volatility regime: volatility-scaled momentum
    high_vol_mask = df['volatility_regime'] > high_vol_threshold
    high_vol_signal = (df['mom_3'] + df['mom_10']) / (df['volatility_regime'] + 1e-8)
    
    # Low-volatility regime: price acceleration with volume confirmation
    low_vol_mask = df['volatility_regime'] < low_vol_threshold
    low_vol_signal = (df['mom_accel_3'] + df['mom_accel_10']) * (1 + df['volume_confirmation'])
    
    # Medium-volatility regime: blended approach
    medium_vol_mask = ~(high_vol_mask | low_vol_mask)
    medium_vol_signal = (df['mom_3'] + df['mom_10'] + df['mom_accel_3'] + df['mom_accel_10']) / 4
    
    # Combine signals based on volatility regime
    factor = pd.Series(index=df.index, dtype=float)
    factor[high_vol_mask] = high_vol_signal[high_vol_mask]
    factor[low_vol_mask] = low_vol_signal[low_vol_mask]
    factor[medium_vol_mask] = medium_vol_signal[medium_vol_mask]
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'atr_5', 
                   'daily_range', 'range_vol_5', 'atr_norm', 'range_vol_norm',
                   'volatility_regime', 'mom_3', 'mom_10', 'mom_accel_3', 
                   'mom_accel_10', 'volume_slope', 'price_impact', 
                   'volume_slope_norm', 'price_impact_norm', 'volume_confirmation']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return factor
