import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add small epsilon to avoid division by zero
    eps = 0.001
    
    # Fracture Momentum Structure
    # Micro Fracture
    data['micro_fracture'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + eps)) - \
                            ((data['close'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    # Meso Fracture
    data['high_3d'] = data['high'].rolling(window=4, min_periods=4).max()
    data['low_3d'] = data['low'].rolling(window=4, min_periods=4).min()
    data['high_6d_prev'] = data['high'].shift(3).rolling(window=4, min_periods=4).max()
    data['low_6d_prev'] = data['low'].shift(3).rolling(window=4, min_periods=4).min()
    
    data['meso_fracture'] = ((data['close'] - data['close'].shift(3)) / (data['high_3d'] - data['low_3d'] + eps)) - \
                           ((data['close'].shift(3) - data['close'].shift(6)) / (data['high_6d_prev'] - data['low_6d_prev'] + eps))
    
    # Macro Fracture
    data['high_8d'] = data['high'].rolling(window=9, min_periods=9).max()
    data['low_8d'] = data['low'].rolling(window=9, min_periods=9).min()
    data['high_16d_prev'] = data['high'].shift(8).rolling(window=9, min_periods=9).max()
    data['low_16d_prev'] = data['low'].shift(8).rolling(window=9, min_periods=9).min()
    
    data['macro_fracture'] = ((data['close'] - data['close'].shift(8)) / (data['high_8d'] - data['low_8d'] + eps)) - \
                            ((data['close'].shift(8) - data['close'].shift(16)) / (data['high_16d_prev'] - data['low_16d_prev'] + eps))
    
    # Asymmetric Volume Fracture Dynamics
    # Buy Pressure Fracture
    data['buy_pressure'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'] + eps)
    data['buy_pressure_fracture'] = data['buy_pressure'] - data['buy_pressure'].shift(1)
    
    # Sell Pressure Fracture
    data['sell_pressure'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + eps)
    data['sell_pressure_fracture'] = data['sell_pressure'] - data['sell_pressure'].shift(1)
    
    # Pressure Asymmetry Fracture
    data['pressure_asymmetry_fracture'] = (data['buy_pressure_fracture'] - data['sell_pressure_fracture']) * \
                                         np.sign(data['buy_pressure_fracture'] - data['sell_pressure_fracture'].shift(1))
    
    # Asymmetric Volatility Fracture
    # Up Day Volatility Fracture
    data['up_vol_fracture'] = ((data['high'] - data['open']) / (data['close'] - data['low'] + eps)) - \
                             ((data['high'].shift(1) - data['open'].shift(1)) / (data['close'].shift(1) - data['low'].shift(1) + eps))
    
    # Down Day Volatility Fracture
    data['down_vol_fracture'] = ((data['open'] - data['low']) / (data['high'] - data['close'] + eps)) - \
                               ((data['open'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['close'].shift(1) + eps))
    
    # Volatility Asymmetry Fracture
    data['volatility_asymmetry_fracture'] = data['up_vol_fracture'] - data['down_vol_fracture']
    
    # Fracture Regime Detection
    # Momentum Fracture Alignment
    data['micro_meso_alignment'] = np.sign(data['micro_fracture']) * np.sign(data['meso_fracture']) * \
                                  np.abs(data['micro_fracture'] - data['meso_fracture'])
    
    data['meso_macro_alignment'] = np.sign(data['meso_fracture']) * np.sign(data['macro_fracture']) * \
                                  np.abs(data['meso_fracture'] - data['macro_fracture'])
    
    data['fracture_cascade'] = data['micro_meso_alignment'] * data['meso_macro_alignment'] * np.sign(data['buy_pressure_fracture'])
    
    # Volume-Price Fracture Integration
    data['volume_price_micro'] = data['buy_pressure_fracture'] * data['micro_fracture'] * (data['high'] - data['low']) / data['close']
    data['volume_price_meso'] = data['pressure_asymmetry_fracture'] * data['meso_fracture'] * (data['high_3d'] - data['low_3d']) / data['close']
    data['volume_price_macro'] = data['volatility_asymmetry_fracture'] * data['macro_fracture'] * (data['high_8d'] - data['low_8d']) / data['close']
    
    # Range Fracture Dynamics
    data['range_expansion_fracture'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) - \
                                      ((data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(2) - data['low'].shift(2)))
    
    data['range_fracture_intensity'] = ((data['high_3d'] - data['low_3d']) / (data['high_6d_prev'] - data['low_6d_prev'])) - 1
    
    data['range_regime_fracture'] = ((data['high_8d'] - data['low_8d']) / (data['high_16d_prev'] - data['low_16d_prev'])) - 1
    
    # Asymmetric Fracture Enhancement
    data['volatility_enhanced_fracture'] = (data['micro_fracture'] - data['meso_fracture']) * data['volatility_asymmetry_fracture']
    data['volume_enhanced_fracture'] = (data['meso_fracture'] - data['macro_fracture']) * data['pressure_asymmetry_fracture']
    data['range_enhanced_fracture'] = (data['micro_fracture'] * data['meso_fracture'] * data['macro_fracture']) * data['range_expansion_fracture']
    
    # Fracture Persistence Framework
    # Momentum Fracture Persistence
    data['micro_sign'] = np.sign(data['micro_fracture'])
    data['micro_persistence'] = data['micro_sign'].rolling(window=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) / 3 if len(x) == 3 else np.nan, raw=False
    )
    
    data['meso_sign'] = np.sign(data['meso_fracture'])
    data['meso_persistence'] = data['meso_sign'].rolling(window=4, step=3).apply(
        lambda x: np.sum(x.iloc[-1] == x.iloc[0]) / 2 if len(x) == 4 else np.nan, raw=False
    )
    
    data['macro_sign'] = np.sign(data['macro_fracture'])
    data['macro_persistence'] = data['macro_sign'].rolling(window=9, step=8).apply(
        lambda x: np.sum(x.iloc[-1] == x.iloc[0]) / 2 if len(x) == 9 else np.nan, raw=False
    )
    
    # Volume Fracture Consistency
    data['buy_pressure_sign'] = np.sign(data['buy_pressure_fracture'])
    data['buy_pressure_persistence'] = data['buy_pressure_sign'].rolling(window=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) / 3 if len(x) == 3 else np.nan, raw=False
    )
    
    data['pressure_asymmetry_sign'] = np.sign(data['pressure_asymmetry_fracture'])
    data['pressure_asymmetry_persistence'] = data['pressure_asymmetry_sign'].rolling(window=4, step=3).apply(
        lambda x: np.sum(x.iloc[-1] == x.iloc[0]) / 2 if len(x) == 4 else np.nan, raw=False
    )
    
    data['volatility_asymmetry_sign'] = np.sign(data['volatility_asymmetry_fracture'])
    data['volatility_asymmetry_persistence'] = data['volatility_asymmetry_sign'].rolling(window=9, step=8).apply(
        lambda x: np.sum(x.iloc[-1] == x.iloc[0]) / 2 if len(x) == 9 else np.nan, raw=False
    )
    
    # Range Fracture Stability
    data['range_expansion_sign'] = np.sign(data['range_expansion_fracture'])
    data['range_expansion_persistence'] = data['range_expansion_sign'].rolling(window=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) / 3 if len(x) == 3 else np.nan, raw=False
    )
    
    data['range_intensity_sign'] = np.sign(data['range_fracture_intensity'])
    data['range_intensity_persistence'] = data['range_intensity_sign'].rolling(window=4, step=3).apply(
        lambda x: np.sum(x.iloc[-1] == x.iloc[0]) / 2 if len(x) == 4 else np.nan, raw=False
    )
    
    data['range_regime_sign'] = np.sign(data['range_regime_fracture'])
    data['range_regime_persistence'] = data['range_regime_sign'].rolling(window=9, step=8).apply(
        lambda x: np.sum(x.iloc[-1] == x.iloc[0]) / 2 if len(x) == 9 else np.nan, raw=False
    )
    
    # Final Asymmetric Fracture Alpha Construction
    data['fracture_momentum_core'] = (data['micro_fracture'] + data['meso_fracture'] + data['macro_fracture']) * \
                                    np.sign(data['volume_price_micro']) * np.sign(data['volume_price_meso'])
    
    data['asymmetric_fracture_core'] = data['volatility_enhanced_fracture'] * data['volume_enhanced_fracture'] * data['range_enhanced_fracture']
    
    data['fracture_persistence_core'] = data['micro_persistence'] * data['buy_pressure_persistence'] * data['range_expansion_persistence']
    
    # Multi-Scale Asymmetric Fracture Alpha
    data['alpha'] = data['fracture_momentum_core'] * data['asymmetric_fracture_core'] * \
                   data['fracture_persistence_core'] * data['fracture_cascade']
    
    return data['alpha']
