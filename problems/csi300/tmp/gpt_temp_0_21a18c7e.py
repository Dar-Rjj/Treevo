import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Components
    df['true_range_efficiency'] = (df['high'] - df['low']) / (abs(df['close'] - df['close'].shift(1)) + 1e-8)
    df['opening_gap_efficiency'] = abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    df['closing_momentum_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Multi-Scale Momentum Analysis
    df['short_term_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['medium_term_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['long_term_momentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Fracture Intensity Measurement
    df['price_fracture'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) - 1
    df['volume_fracture'] = df['volume'] / df['volume'].shift(1) - 1
    df['gap_fracture'] = abs(df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    
    # Efficiency-Weighted Momentum
    df['short_efficiency_momentum'] = df['short_term_momentum'] * (1 - df['opening_gap_efficiency'])
    df['medium_efficiency_momentum'] = df['medium_term_momentum'] * df['closing_momentum_efficiency']
    df['long_efficiency_momentum'] = df['long_term_momentum'] / (df['true_range_efficiency'] + 1e-8)
    
    # Volume Asymmetry Analysis
    df['bullish_volume_pressure'] = df['volume'] * np.maximum(0, df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['bearish_volume_pressure'] = df['volume'] * np.maximum(0, df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8)
    df['volume_imbalance'] = (df['bullish_volume_pressure'] - df['bearish_volume_pressure']) / (df['bullish_volume_pressure'] + df['bearish_volume_pressure'] + 1e-8)
    
    # Fracture-Enhanced Signals
    conditions = [
        df['price_fracture'] > 0.03,
        df['price_fracture'] < 0.01
    ]
    choices = [
        df['short_efficiency_momentum'] * (1 + abs(df['price_fracture'])),
        df['long_efficiency_momentum'] / (1 + abs(df['price_fracture']))
    ]
    df['fracture_signal'] = np.select(conditions, choices, default=df['medium_efficiency_momentum'] * abs(df['price_fracture']))
    
    # Volume Confidence Assessment
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(lambda x: (x > x.shift(1)).sum(), raw=False)
    df['volume_trend'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_validation'] = df['volume_imbalance'] * np.sign(df['volume_trend'])
    
    # Regime-Adaptive Processing
    df['avg_range_9'] = (df['high'] - df['low']).rolling(window=9).mean()
    conditions_regime = [
        (df['high'] - df['low']) > df['avg_range_9'],
        (df['high'] - df['low']) < df['avg_range_9']
    ]
    choices_regime = [
        df['fracture_signal'] * df['volume_validation'],
        df['fracture_signal'] * df['volume_validation']
    ]
    df['regime_signal'] = np.select(conditions_regime, choices_regime, default=df['fracture_signal'] * df['volume_validation'])
    
    # Reversal Detection Enhancement
    df['position_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['overbought'] = df['position_ratio'] > 0.8
    df['oversold'] = df['position_ratio'] < 0.2
    df['volume_divergence'] = (df['volume'] > df['volume'].shift(1)) & (df['close'] < df['close'].shift(1))
    
    conditions_reversal = [
        df['overbought'] & df['volume_divergence'],
        df['oversold'] & df['volume_divergence']
    ]
    choices_reversal = [-1, 1]
    df['reversal_multiplier'] = np.select(conditions_reversal, choices_reversal, default=0)
    
    # Composite Alpha Generation
    df['base_signal'] = df['regime_signal']
    df['reversal_adjusted'] = df['base_signal'] * (1 + df['reversal_multiplier'])
    df['final_alpha'] = df['reversal_adjusted'] * df['volume_persistence'] / 5
    
    return df['final_alpha']
