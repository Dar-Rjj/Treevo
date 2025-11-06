import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper function for rolling percentiles
    def rolling_percentile(series, window, percentile):
        return series.rolling(window).apply(lambda x: np.percentile(x, percentile), raw=True)
    
    # 1. Fractal Asymmetric Momentum Framework
    # Multi-Scale Asymmetric Momentum
    data['micro_asym_mom'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    data['high_2d'] = data['high'].rolling(window=3).max()  # t-2 to t
    data['low_2d'] = data['low'].rolling(window=3).min()
    data['meso_asym_mom'] = (data['high_2d'] - data['open']) - (data['open'] - data['low_2d'])
    
    data['high_5d'] = data['high'].rolling(window=6).max()  # t-5 to t
    data['low_5d'] = data['low'].rolling(window=6).min()
    data['macro_asym_mom'] = (data['high_5d'] - data['open']) - (data['open'] - data['low_5d'])
    
    data['fractal_asym_cascade'] = data['micro_asym_mom'] * data['meso_asym_mom'] * data['macro_asym_mom']
    
    # Volume Fractal Asymmetry
    data['volume_micro_asym'] = ((data['high'] - data['open']) - (data['close'] - data['low'])) / (data['volume'] + 0.001)
    data['volume_meso_asym'] = ((data['high_2d'] - data['open']) - (data['close'] - data['low_2d'])) / (data['volume'] + 0.001)
    data['volume_macro_asym'] = ((data['high_5d'] - data['open']) - (data['close'] - data['low_5d'])) / (data['volume'] + 0.001)
    data['volume_fractal_asym_cascade'] = data['volume_micro_asym'] * data['volume_meso_asym'] * data['volume_macro_asym']
    
    # 2. Efficiency-Compression Fractal Dynamics
    # Fractal Range Efficiency
    data['micro_range_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['meso_range_eff'] = (data['close'] - data['open']) / (data['high_2d'] - data['low_2d'] + 0.001)
    data['macro_range_eff'] = (data['close'] - data['open']) / (data['high_5d'] - data['low_5d'] + 0.001)
    data['fractal_eff_cascade'] = data['micro_range_eff'] * data['meso_range_eff'] * data['macro_range_eff']
    
    # Fractal Volatility Compression
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    
    # Micro Compression
    micro_comp_cond1 = data['true_range'] < rolling_percentile(data['true_range'], 6, 30)  # t-5 to t
    micro_comp_cond2 = data['true_range'] > rolling_percentile(data['true_range'], 6, 70)
    data['micro_comp'] = np.where(micro_comp_cond1, 1, np.where(micro_comp_cond2, -1, 0))
    
    # Meso Compression
    meso_comp_cond1 = data['true_range'] < rolling_percentile(data['true_range'], 11, 30)  # t-10 to t
    meso_comp_cond2 = data['true_range'] > rolling_percentile(data['true_range'], 11, 70)
    data['meso_comp'] = np.where(meso_comp_cond1, 1, np.where(meso_comp_cond2, -1, 0))
    
    # Macro Compression
    macro_comp_cond1 = data['true_range'] < rolling_percentile(data['true_range'], 21, 30)  # t-20 to t
    macro_comp_cond2 = data['true_range'] > rolling_percentile(data['true_range'], 21, 70)
    data['macro_comp'] = np.where(macro_comp_cond1, 1, np.where(macro_comp_cond2, -1, 0))
    
    data['fractal_comp_cascade'] = (data['micro_comp'] - data['meso_comp']) * (data['meso_comp'] - data['macro_comp'])
    
    # 3. Regime Momentum Asymmetric Persistence
    # Asymmetric Regime Persistence
    daily_asym = (data['high'] - data['open']) > (data['open'] - data['low'])
    
    data['short_asym_pers'] = daily_asym.rolling(window=2).sum() - (~daily_asym).rolling(window=2).sum()
    data['medium_asym_pers'] = daily_asym.rolling(window=5).sum() - (~daily_asym).rolling(window=5).sum()
    data['long_asym_pers'] = daily_asym.rolling(window=13).sum() - (~daily_asym).rolling(window=13).sum()
    data['asym_regime_pers_cascade'] = data['short_asym_pers'] * data['medium_asym_pers'] * data['long_asym_pers']
    
    # Volume Asymmetric Persistence
    vol_short_up = (data['volume'] > data['volume'].shift(1)).rolling(window=2).sum()
    vol_short_down = (data['volume'] < data['volume'].shift(1)).rolling(window=2).sum()
    data['volume_short_asym'] = (vol_short_up - vol_short_down) * np.sign(data['micro_asym_mom'])
    
    vol_med_up = (data['volume'] > data['volume'].shift(3)).rolling(window=5).sum()
    vol_med_down = (data['volume'] < data['volume'].shift(3)).rolling(window=5).sum()
    data['volume_medium_asym'] = (vol_med_up - vol_med_down) * np.sign(data['micro_asym_mom'])
    
    vol_long_up = (data['volume'] > data['volume'].shift(8)).rolling(window=13).sum()
    vol_long_down = (data['volume'] < data['volume'].shift(8)).rolling(window=13).sum()
    data['volume_long_asym'] = (vol_long_up - vol_long_down) * np.sign(data['micro_asym_mom'])
    
    data['volume_asym_pers_cascade'] = data['volume_short_asym'] * data['volume_medium_asym'] * data['volume_long_asym']
    
    # 4. Composite Alpha Generation
    # Core Fractal Asymmetric Components
    data['fractal_asym_core'] = data['fractal_asym_cascade'] * data['volume_fractal_asym_cascade']
    data['asym_regime_core'] = data['asym_regime_pers_cascade'] * data['volume_asym_pers_cascade']
    data['efficiency_compression_core'] = data['fractal_eff_cascade'] * data['fractal_comp_cascade']
    
    # Multi-Scale Integration
    data['base_asym_factor'] = data['fractal_asym_core'] * data['asym_regime_core'] * data['efficiency_compression_core']
    data['scale_weighted_asym'] = data['base_asym_factor'] * data['micro_asym_mom'] * data['meso_asym_mom'] * data['macro_asym_mom']
    
    # Final Alpha Composition
    data['final_alpha'] = data['scale_weighted_asym'] * data['fractal_asym_cascade']
    
    return data['final_alpha']
