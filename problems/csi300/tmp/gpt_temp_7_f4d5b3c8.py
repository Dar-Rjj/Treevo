import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Scale Fractal Dimension Analysis
    data['micro_fractal'] = (data['high'] - data['low']) / (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min() + epsilon)
    data['meso_fractal'] = (data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min()) / (data['high'].rolling(window=11).max() - data['low'].rolling(window=11).min() + epsilon)
    data['macro_fractal'] = (data['high'].rolling(window=21).max() - data['low'].rolling(window=21).min()) / (data['high'].rolling(window=41).max() - data['low'].rolling(window=41).min() + epsilon)
    data['volume_fractal'] = data['volume'] / (data['volume'].rolling(window=6).mean() + epsilon)
    
    # Fractal Regime
    conditions = [
        data['micro_fractal'] > 1,
        data['meso_fractal'] > 1
    ]
    choices = ['High', 'Medium']
    data['fractal_regime_base'] = np.select(conditions, choices, default='Low')
    data['volume_regime'] = np.where(data['volume_fractal'] > 1, '_Expanding', '_Contracting')
    data['fractal_regime'] = data['fractal_regime_base'] + data['volume_regime']
    
    # Fractal-Adaptive Trend Construction
    data['high_fractal_trend'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + epsilon)
    data['medium_fractal_trend'] = (data['close'] - data['close'].rolling(window=6).mean()) / (data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min() + epsilon)
    data['low_fractal_trend'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(window=21).max() - data['low'].rolling(window=21).min() + epsilon)
    
    # Regime Trend
    regime_conditions = [
        data['fractal_regime'].isin(['High_Expanding', 'Medium_Expanding']),
        data['fractal_regime'].isin(['Low_Expanding', 'High_Contracting']),
        data['fractal_regime'].isin(['Medium_Contracting', 'Low_Contracting'])
    ]
    regime_choices = [
        data['high_fractal_trend'],
        data['medium_fractal_trend'],
        data['low_fractal_trend']
    ]
    data['regime_trend'] = np.select(regime_conditions, regime_choices)
    
    # Bid-Ask Pressure Asymmetry
    data['opening_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['closing_pressure'] = (data['close'] - data['open']) / (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min() + epsilon)
    data['intraday_imbalance'] = ((data['high'] - data['close']) - (data['close'] - data['low'])) / (data['high'] - data['low'] + epsilon)
    data['pressure_asymmetry'] = data['opening_pressure'] * data['closing_pressure'] * data['intraday_imbalance']
    
    # Volume-Price Fractal Divergence
    data['volume_acceleration'] = data['volume'] / (data['volume'].shift(1) + epsilon) - 1
    data['price_return'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + epsilon)
    data['price_volume_fractal'] = data['price_return'] - data['volume_acceleration']
    data['divergence_intensity'] = np.abs(data['price_volume_fractal']) / (data['close'].rolling(window=6).std() / data['close'].rolling(window=6).mean() + epsilon)
    data['fractal_signal'] = np.sign(data['price_volume_fractal']) * data['divergence_intensity']
    
    # Multi-Fractal Trend Alignment
    data['micro_trend'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + epsilon)
    data['meso_trend'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + epsilon)
    data['macro_trend'] = (data['close'] - data['close'].shift(20)) / (data['close'].shift(20) + epsilon)
    data['fractal_alignment'] = (np.sign(data['micro_trend']) * np.sign(data['meso_trend']) * np.sign(data['macro_trend']) * 
                                (np.abs(data['micro_trend']) + np.abs(data['meso_trend']) + np.abs(data['macro_trend'])))
    
    # Fractal-Adaptive Factor Construction
    data['high_fractal_core'] = data['regime_trend'] * data['pressure_asymmetry'] * data['fractal_signal']
    data['medium_fractal_core'] = data['regime_trend'] * data['fractal_alignment'] * data['pressure_asymmetry']
    data['low_fractal_core'] = data['regime_trend'] * data['fractal_alignment'] * data['fractal_signal']
    
    # Adaptive Core
    core_conditions = [
        data['fractal_regime'].isin(['High_Expanding', 'Medium_Expanding']),
        data['fractal_regime'].isin(['Low_Expanding', 'High_Contracting']),
        data['fractal_regime'].isin(['Medium_Contracting', 'Low_Contracting'])
    ]
    core_choices = [
        data['high_fractal_core'],
        data['medium_fractal_core'],
        data['low_fractal_core']
    ]
    data['adaptive_core'] = np.select(core_conditions, core_choices)
    
    # Fractal Persistence Enhancement
    data['trend_persistence'] = (data['regime_trend'].rolling(window=6).apply(
        lambda x: np.sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) / 5 if len(x) == 6 else np.nan, raw=False))
    data['pressure_persistence'] = (data['pressure_asymmetry'].rolling(window=4).apply(
        lambda x: np.sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) / 3 if len(x) == 4 else np.nan, raw=False))
    data['signal_persistence'] = (data['fractal_signal'].rolling(window=6).apply(
        lambda x: np.sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) / 5 if len(x) == 6 else np.nan, raw=False))
    data['fractal_quality'] = data['trend_persistence'] * data['pressure_persistence'] * data['signal_persistence']
    
    # Volume-Weighted Fractal Momentum
    data['trade_intensity'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + epsilon)
    
    # Calculate price changes and volume-weighted changes
    price_changes = data['close'] - data['close'].shift(1)
    volume_weighted_changes = price_changes * data['volume']
    data['volume_weighted_trend'] = volume_weighted_changes / (volume_weighted_changes.rolling(window=6).mean() + epsilon)
    data['volume_adjustment'] = data['trade_intensity'] * data['volume_weighted_trend']
    
    # Final Alpha Construction
    data['core_fractor'] = data['adaptive_core'] * data['fractal_quality']
    data['volume_enhancement'] = data['core_fractor'] * data['volume_adjustment']
    data['fractal_confirmation'] = data['volume_enhancement'] * np.sign(data['regime_trend']) * np.sign(data['fractal_alignment'])
    data['final_alpha'] = data['core_fractor'] * data['volume_enhancement'] * data['fractal_confirmation']
    
    return data['final_alpha']
