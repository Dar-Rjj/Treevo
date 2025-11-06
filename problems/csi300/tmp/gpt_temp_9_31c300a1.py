import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Momentum Construction
    df['price_momentum_spectrum'] = (df['close'] / df['close'].shift(2) - 1) - (df['close'] / df['close'].shift(8) - 1)
    df['volume_momentum_spectrum'] = (df['volume'] / df['volume'].shift(2) - 1) - (df['volume'] / df['volume'].shift(8) - 1)
    df['momentum_coherence'] = np.sign(df['price_momentum_spectrum']) * np.sign(df['volume_momentum_spectrum'])
    
    # Fractal Compression Dynamics
    df['tr5'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(4)),
            np.abs(df['low'] - df['close'].shift(4))
        )
    )
    df['fractal_range_compression'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    df['compression_momentum'] = df['fractal_range_compression'] * df['price_momentum_spectrum']
    
    # Microstructure Pressure Integration
    df['depth_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['execution_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Volume Pressure Asymmetry
    close_gt_open = df['close'] > df['open']
    close_lt_open = df['close'] < df['open']
    
    vol_positive = df['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[close_gt_open.iloc[-5:].values]) if len(x) == 5 else np.nan, raw=False
    )
    vol_negative = df['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[close_lt_open.iloc[-5:].values]) if len(x) == 5 else np.nan, raw=False
    )
    vol_total = df['volume'].rolling(window=5).sum()
    df['volume_pressure_asymmetry'] = (vol_positive - vol_negative) / vol_total
    
    # Fractal-Pressure Divergence
    df['fractal_efficiency'] = ((df['close'] - df['open']) / df['tr5']) * (
        (df['high'] / df['high'].shift(2) - 1) - (df['low'] / df['low'].shift(2) - 1)
    )
    df['fractal_gap_pressure'] = (df['open'] / df['close'].shift(1) - 1) * df['fractal_range_compression']
    df['volume_compression_alignment'] = df['compression_momentum'] * np.sign(df['volume'] - df['volume'].shift(1))
    
    # Multi-Scale Breakout Dynamics
    df['rolling_max_high_5'] = df['high'].rolling(window=5).max()
    df['compression_breakout'] = (df['close'] - df['rolling_max_high_5']) * df['fractal_range_compression']
    
    df['volume_avg_5'] = df['volume'].rolling(window=5).mean()
    df['volume_confirmed_breakout'] = df['compression_breakout'] * (df['volume'] / df['volume_avg_5'])
    
    df['tr1'] = df['high'] - df['low']
    df['range_momentum'] = (df['tr5'] / df['tr5'].shift(2) - 1) - (df['tr1'] / df['tr1'].shift(2) - 1)
    
    # Adaptive Factor Synthesis
    df['core_momentum_pressure'] = (
        df['fractal_efficiency'] * df['fractal_gap_pressure'] * df['volume_compression_alignment'] * 
        df['depth_pressure'] * df['execution_efficiency'] * df['volume_pressure_asymmetry'] * df['momentum_coherence']
    )
    
    df['breakout_integration'] = (
        df['core_momentum_pressure'] * df['volume_confirmed_breakout'] * df['range_momentum']
    )
    
    df['final_factor'] = (
        df['breakout_integration'] * df['compression_momentum'] * 
        (1 + np.abs(df['volume_pressure_asymmetry'])) * df['execution_efficiency']
    )
    
    return df['final_factor']
