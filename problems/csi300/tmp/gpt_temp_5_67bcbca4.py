import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Reversal Dynamics
    # Micro-Fractal Reversal
    data['micro_fractal'] = (data['close'] - data['low']) / (data['high'] - data['low']) - \
                           (data['high'] - data['close']) / (data['high'] - data['low'])
    
    # Meso-Fractal Reversal
    data['min_low_5'] = data['low'].rolling(window=5, min_periods=1).min()
    data['max_high_5'] = data['high'].rolling(window=5, min_periods=1).max()
    data['meso_fractal'] = (data['close'] - data['min_low_5']) / (data['max_high_5'] - data['min_low_5']) - 0.5
    
    # Macro-Fractal Reversal
    data['macro_fractal'] = (data['close'] - data['close'].shift(10)) / \
                           (data['high'].shift(10) - data['low'].shift(10))
    data['macro_fractal'] = data['macro_fractal'].fillna(0)
    
    # Fractal Compression Analysis
    # Range Compression
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['range_compression'] = data['range_compression'].fillna(1)
    
    # Compression Intensity
    data['daily_range_current'] = abs(data['close'] - data['open'])
    data['daily_range_prev'] = abs(data['close'].shift(1) - data['open'].shift(1))
    data['daily_range_prev'] = data['daily_range_prev'].fillna(1)
    data['compression_intensity'] = data['range_compression'] * (data['daily_range_current'] / data['daily_range_prev'])
    
    # Volume-Intensity Confirmation
    # Intensity Ratio
    data['intensity_2d'] = abs(data['close'] - data['open']).rolling(window=3, min_periods=1).sum()
    data['intensity_5d'] = abs(data['close'].shift(3) - data['open'].shift(3)).rolling(window=5, min_periods=1).sum()
    data['intensity_ratio'] = data['intensity_2d'] / data['intensity_5d']
    data['intensity_ratio'] = data['intensity_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Intensity Divergence
    data['volume_4d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_intensity_div'] = data['intensity_ratio'] * (data['volume'] / data['volume_4d_avg'])
    
    # Gap-Fill Momentum System
    # Gap Absorption Efficiency
    data['gap_absorption'] = (abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))) * \
                            np.sign(data['close'] - data['open'])
    data['gap_absorption'] = data['gap_absorption'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Gap-Momentum Factor
    data['price_change_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['momentum_count'] = data['price_change_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0
    )
    data['gap_momentum'] = np.sqrt(data['gap_absorption'] * (data['momentum_count'] / 5))
    data['gap_momentum'] = data['gap_momentum'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Multi-Scale Divergence Framework
    # Fractal Reversal Divergence
    data['fractal_reversal_div'] = data['micro_fractal'] * (data['meso_fractal'] - data['macro_fractal'])
    
    # Reversal-Compression Divergence
    data['reversal_compression_div'] = data['fractal_reversal_div'] * (1 - data['compression_intensity'])
    
    # Fractal Regime Classification
    data['volume_ratio'] = data['volume'] / data['volume_4d_avg']
    
    data['reversal_regime'] = (data['fractal_reversal_div'] > 0.2) & (data['volume_ratio'] > 1.2)
    data['compression_regime'] = data['compression_intensity'] < 0.6
    data['normal_regime'] = ~data['reversal_regime'] & ~data['compression_regime']
    
    # Breakout Enhancement
    # Fractal Breakout Component
    data['min_low_9'] = data['low'].rolling(window=10, min_periods=1).min()
    data['max_high_9'] = data['high'].rolling(window=10, min_periods=1).max()
    data['fractal_breakout'] = (data['close'] - data['min_low_9']) / (data['max_high_9'] - data['min_low_9'])
    data['fractal_breakout'] = data['fractal_breakout'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Enhanced Breakout
    data['gap_ratio'] = abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['gap_ratio'] = data['gap_ratio'].fillna(0)
    data['enhanced_breakout'] = data['fractal_breakout'] * (1 + (data['gap_ratio'] > 0.1).astype(float))
    
    # Final Alpha Synthesis
    # Regime-Adaptive Selection
    data['selected_alpha'] = 0.0
    
    # Reversal Regime
    reversal_mask = data['reversal_regime']
    data.loc[reversal_mask, 'selected_alpha'] = data.loc[reversal_mask, 'reversal_compression_div'] * \
                                               data.loc[reversal_mask, 'volume_intensity_div'] * \
                                               data.loc[reversal_mask, 'enhanced_breakout']
    
    # Compression Regime
    compression_mask = data['compression_regime']
    data.loc[compression_mask, 'selected_alpha'] = data.loc[compression_mask, 'volume_intensity_div'] * \
                                                  data.loc[compression_mask, 'compression_intensity']
    
    # Normal Regime
    normal_mask = data['normal_regime']
    data.loc[normal_mask, 'selected_alpha'] = (data.loc[normal_mask, 'reversal_compression_div'] + \
                                              data.loc[normal_mask, 'gap_momentum']) / 2
    
    # Multi-Scale Fractal Reversal-Compression Alpha
    data['range_sum_4d'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).sum()
    data['range_sum_9d'] = (data['high'] - data['low']).rolling(window=10, min_periods=1).sum()
    data['range_ratio'] = data['range_sum_4d'] / data['range_sum_9d']
    data['range_ratio'] = data['range_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Final alpha factor
    alpha = data['selected_alpha'] * data['range_ratio']
    
    return alpha
