import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Fractal Asymmetric Momentum
    # Micro Asymmetric Momentum
    data['micro_asym_mom'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Meso Asymmetric Momentum
    data['high_2d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_2d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['meso_asym_mom'] = (data['high_2d'] - data['open']) - (data['open'] - data['low_2d'])
    
    # Macro Asymmetric Momentum
    data['high_5d'] = data['high'].rolling(window=6, min_periods=1).max()
    data['low_5d'] = data['low'].rolling(window=6, min_periods=1).min()
    data['macro_asym_mom'] = (data['high_5d'] - data['open']) - (data['open'] - data['low_5d'])
    
    # Fractal Asymmetric Cascade
    data['fractal_asym_cascade'] = data['micro_asym_mom'] * data['meso_asym_mom'] * data['macro_asym_mom']
    
    # Volume Asymmetric Dynamics
    # Volume Micro Asymmetry
    data['volume_micro_asym'] = ((data['high'] - data['open']) - (data['close'] - data['low'])) / (data['volume'] + 0.001)
    
    # Volume Meso Asymmetry
    data['volume_meso_asym'] = ((data['high_2d'] - data['open']) - (data['close'] - data['low_2d'])) / (data['volume'] + 0.001)
    
    # Volume Macro Asymmetry
    data['volume_macro_asym'] = ((data['high_5d'] - data['open']) - (data['close'] - data['low_5d'])) / (data['volume'] + 0.001)
    
    # Volume Asymmetric Cascade
    data['volume_asym_cascade'] = data['volume_micro_asym'] * data['volume_meso_asym'] * data['volume_macro_asym']
    
    # Fractal Range Efficiency
    # Micro Range Efficiency
    data['micro_range_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Meso Range Efficiency
    data['meso_range_eff'] = (data['close'] - data['open']) / (data['high_2d'] - data['low_2d'] + 0.001)
    
    # Macro Range Efficiency
    data['macro_range_eff'] = (data['close'] - data['open']) / (data['high_5d'] - data['low_5d'] + 0.001)
    
    # Fractal Efficiency Cascade
    data['fractal_eff_cascade'] = data['micro_range_eff'] * data['meso_range_eff'] * data['macro_range_eff']
    
    # Asymmetric Persistence
    # Calculate daily asymmetric momentum for persistence counting
    data['daily_asym'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Short Asymmetric Persistence (2 days)
    def count_asymmetry_2d(series):
        if len(series) < 3:
            return 0
        pos_count = sum(1 for x in series[-3:-1] if x > 0)
        neg_count = sum(1 for x in series[-3:-1] if x < 0)
        return pos_count - neg_count
    
    # Medium Asymmetric Persistence (5 days)
    def count_asymmetry_5d(series):
        if len(series) < 6:
            return 0
        pos_count = sum(1 for x in series[-6:-1] if x > 0)
        neg_count = sum(1 for x in series[-6:-1] if x < 0)
        return pos_count - neg_count
    
    # Long Asymmetric Persistence (13 days)
    def count_asymmetry_13d(series):
        if len(series) < 14:
            return 0
        pos_count = sum(1 for x in series[-14:-1] if x > 0)
        neg_count = sum(1 for x in series[-14:-1] if x < 0)
        return pos_count - neg_count
    
    # Calculate rolling persistence
    data['short_asym_pers'] = data['daily_asym'].rolling(window=3, min_periods=1).apply(
        lambda x: count_asymmetry_2d(x), raw=False
    )
    data['medium_asym_pers'] = data['daily_asym'].rolling(window=6, min_periods=1).apply(
        lambda x: count_asymmetry_5d(x), raw=False
    )
    data['long_asym_pers'] = data['daily_asym'].rolling(window=14, min_periods=1).apply(
        lambda x: count_asymmetry_13d(x), raw=False
    )
    
    # Asymmetric Persistence Cascade
    data['asym_pers_cascade'] = data['short_asym_pers'] * data['medium_asym_pers'] * data['long_asym_pers']
    
    # Composite Alpha Generation
    # Core Components
    data['asymmetric_core'] = data['fractal_asym_cascade'] * data['volume_asym_cascade']
    data['persistence_core'] = data['asym_pers_cascade']
    data['efficiency_core'] = data['fractal_eff_cascade']
    
    # Final Alpha
    alpha_factor = data['asymmetric_core'] * data['persistence_core'] * data['efficiency_core']
    
    return alpha_factor
