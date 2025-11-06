import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Sign Blend
    df['mom_3d'] = np.sign(df['close'] / df['close'].shift(3) - 1)
    df['mom_8d'] = np.sign(df['close'] / df['close'].shift(8) - 1)
    df['mom_21d'] = np.sign(df['close'] / df['close'].shift(21) - 1)
    df['momentum_sign_product'] = df['mom_3d'] * df['mom_8d'] * df['mom_21d']
    
    # Adaptive Volume Acceleration
    df['vol_mom_5d'] = df['volume'] / df['volume'].shift(5)
    df['vol_mom_10d'] = df['volume'] / df['volume'].shift(10)
    df['raw_acceleration'] = df['vol_mom_5d'] / df['vol_mom_10d']
    
    # Initialize adaptive EMA
    df['adaptive_ema'] = np.nan
    df.loc[df.index[0], 'adaptive_ema'] = df.loc[df.index[0], 'raw_acceleration']
    
    # Calculate adaptive EMA iteratively
    for i in range(1, len(df)):
        prev_ema = df.iloc[i-1]['adaptive_ema']
        raw_acc = df.iloc[i]['raw_acceleration']
        adaptive_smoothing = 1 / (1 + abs(raw_acc))
        df.iloc[i, df.columns.get_loc('adaptive_ema')] = prev_ema + adaptive_smoothing * (raw_acc - prev_ema)
    
    # Volatility Scaling with True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['inv_vol_scale'] = 1 / df['true_range']
    
    # Time-Decay Emphasis
    df['recent_return_emphasis'] = (df['close'] / df['close'].shift(1)) ** 2
    
    # Regime-Aware Integration
    df['momentum_volume_core'] = df['momentum_sign_product'] * df['adaptive_ema']
    df['vol_adjusted'] = df['momentum_volume_core'] * df['inv_vol_scale']
    df['time_emphasized'] = df['vol_adjusted'] * df['recent_return_emphasis']
    
    # Regime filter (all momentum signs agree)
    df['regime_filter'] = (abs(df['momentum_sign_product']) == 1).astype(int)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['time_emphasized'] * df['regime_filter']
    
    return df['alpha_factor']
