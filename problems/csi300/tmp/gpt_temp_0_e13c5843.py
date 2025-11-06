import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Microstructure Anchoring Framework
    # Price-Level Anchoring Detection
    data['Opening_Anchor_Strength'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Intraday Anchor Persistence
    def intraday_anchor_persistence(row, window=5):
        current_idx = data.index.get_loc(row.name)
        if current_idx < window - 1:
            return np.nan
        count = 0
        for i in range(current_idx - window + 1, current_idx + 1):
            if i < 0:
                continue
            row_data = data.iloc[i]
            range_25p = 0.25 * (row_data['high'] - row_data['low'])
            if abs(row_data['close'] - row_data['open']) <= range_25p:
                count += 1
        return count
    
    data['Intraday_Anchor_Persistence'] = data.apply(intraday_anchor_persistence, axis=1)
    
    # Anchor Breakout Efficiency
    mask = np.abs(data['open'] - data['close'].shift(1)) > 0
    data['Anchor_Breakout_Efficiency'] = np.where(
        mask, 
        (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1)), 
        np.nan
    )
    
    # Volume Anchoring Patterns
    data['Opening_Volume_Concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).sum()
    data['Anchor_Volume_Divergence'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Volume Anchor Persistence
    data['Volume_Anchor_Persistence'] = (
        (data['volume'] > data['volume'].shift(1)).astype(int) + 
        (data['volume'].shift(1) > data['volume'].shift(2)).astype(int)
    )
    
    # Microstructure Regime Classification
    data['Tight_Anchoring_Regime'] = (data['high'] - data['low']) < 0.5 * (data['high'].shift(1) - data['low'].shift(1))
    data['Loose_Anchoring_Regime'] = (data['high'] - data['low']) > 1.5 * (data['high'].shift(1) - data['low'].shift(1))
    data['Normal_Anchoring_Regime'] = ~(data['Tight_Anchoring_Regime'] | data['Loose_Anchoring_Regime'])
    
    # Fractal Momentum System
    # Multi-Scale Fractal Analysis
    data['Micro_Fractal_Momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Meso-Fractal Momentum
    data['Meso_High_Roll'] = data['high'].rolling(window=3, min_periods=1).max()
    data['Meso_Low_Roll'] = data['low'].rolling(window=3, min_periods=1).min()
    data['Meso_Fractal_Momentum'] = (data['close'] - data['close'].shift(2)) / (data['Meso_High_Roll'] - data['Meso_Low_Roll'])
    
    # Macro-Fractal Momentum
    data['Macro_High_Roll'] = data['high'].rolling(window=6, min_periods=1).max()
    data['Macro_Low_Roll'] = data['low'].rolling(window=6, min_periods=1).min()
    data['Macro_Fractal_Momentum'] = (data['close'] - data['close'].shift(5)) / (data['Macro_High_Roll'] - data['Macro_Low_Roll'])
    
    # Fractal Momentum Divergence
    data['Micro_Meso_Divergence'] = data['Micro_Fractal_Momentum'] - data['Meso_Fractal_Momentum']
    data['Meso_Macro_Divergence'] = data['Meso_Fractal_Momentum'] - data['Macro_Fractal_Momentum']
    data['Fractal_Acceleration'] = data['Micro_Meso_Divergence'] - data['Meso_Macro_Divergence']
    
    # Fractal Volume Integration
    data['Volume_Fractal_Alignment'] = np.sign(data['Micro_Fractal_Momentum']) * np.sign(data['Anchor_Volume_Divergence'])
    data['Fractal_Volume_Momentum'] = data['Micro_Fractal_Momentum'] * data['Anchor_Volume_Divergence']
    data['Multi_Scale_Volume_Confirmation'] = data['Meso_Fractal_Momentum'] * data['Volume_Anchor_Persistence']
    
    # Dynamic Regime Transition Framework
    # Volatility Transition Detection
    data['Volatility_Expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1
    data['Volatility_Contraction'] = 1 - (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['Volatility_Regime_Change'] = np.sign(data['Volatility_Expansion']) != np.sign(data['Volatility_Expansion'].shift(1))
    
    # Volume Transition Patterns
    data['Volume_Regime_Break'] = (data['volume'] / data['volume'].shift(1) > 2) | (data['volume'] / data['volume'].shift(1) < 0.5)
    data['Volume_Momentum_Shift'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['Transition_Volume_Confirmation'] = data['Volume_Regime_Break'].astype(int) * data['Volume_Momentum_Shift']
    
    # Price-Volume Transition Alignment
    data['Transition_Efficiency'] = np.where(
        data['Volatility_Regime_Change'],
        (data['close'] - data['open']) / (data['high'] - data['low']),
        np.nan
    )
    data['Volume_Transition_Momentum'] = data['Transition_Efficiency'] * data['Volume_Momentum_Shift']
    data['Anchored_Transition'] = data['Transition_Efficiency'] * data['Opening_Anchor_Strength']
    
    # Momentum Fractal Efficiency
    # Fractal Efficiency Metrics
    data['Micro_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['Meso_Efficiency'] = np.abs(data['close'] - data['close'].shift(2)) / (data['Meso_High_Roll'] - data['Meso_Low_Roll'])
    data['Macro_Efficiency'] = np.abs(data['close'] - data['close'].shift(5)) / (data['Macro_High_Roll'] - data['Macro_Low_Roll'])
    
    # Efficiency Divergence Patterns
    data['Efficiency_Momentum'] = data['Micro_Efficiency'] - data['Meso_Efficiency']
    
    # Efficiency Persistence
    def efficiency_persistence(row):
        current_idx = data.index.get_loc(row.name)
        if current_idx < 3:
            return np.nan
        count = 0
        for i in range(current_idx - 3, current_idx + 1):
            if i < 0:
                continue
            if data.iloc[i]['Micro_Efficiency'] > data.iloc[i]['Meso_Efficiency']:
                count += 1
        return count
    
    data['Efficiency_Persistence'] = data.apply(efficiency_persistence, axis=1)
    data['Fractal_Efficiency_Alignment'] = data['Efficiency_Momentum'] * data['Fractal_Acceleration']
    
    # Volume-Weighted Efficiency
    data['Volume_Efficiency_Momentum'] = data['Efficiency_Momentum'] * data['Anchor_Volume_Divergence']
    data['Efficiency_Volume_Confirmation'] = data['Micro_Efficiency'] * data['Opening_Volume_Concentration']
    data['Persistence_Weighted_Efficiency'] = data['Efficiency_Momentum'] * data['Volume_Anchor_Persistence']
    
    # Cross-Fractal Momentum Framework
    # Price-Volume Fractal Integration
    data['Fractal_Volume_Momentum'] = data['Micro_Fractal_Momentum'] * data['Volume_Momentum_Shift']
    data['Volume_Fractal_Efficiency'] = data['Micro_Efficiency'] * data['Opening_Volume_Concentration']
    data['Cross_Fractal_Divergence'] = data['Fractal_Volume_Momentum'] - data['Volume_Fractal_Efficiency']
    
    # Anchored Fractal Patterns
    data['Anchor_Fractal_Momentum'] = data['Micro_Fractal_Momentum'] * data['Opening_Anchor_Strength']
    data['Fractal_Anchor_Efficiency'] = data['Anchor_Breakout_Efficiency'] * data['Fractal_Acceleration']
    data['Anchored_Volume_Fractal'] = data['Anchor_Fractal_Momentum'] * data['Volume_Anchor_Persistence']
    
    # Transition-Enhanced Fractals
    data['Transition_Fractal_Momentum'] = data['Micro_Fractal_Momentum'] * data['Volatility_Expansion']
    data['Fractal_Transition_Efficiency'] = data['Transition_Efficiency'] * data['Fractal_Acceleration']
    data['Volume_Transition_Fractal'] = data['Transition_Fractal_Momentum'] * data['Transition_Volume_Confirmation']
    
    # Adaptive Fractal Signal Integration
    # Tight Anchoring Signal Generation
    tight_core = data['Anchor_Fractal_Momentum'] * data['Fractal_Anchor_Efficiency']
    data['Tight_Signal'] = tight_core * data['Volume_Anchor_Persistence'] * data['Micro_Efficiency']
    
    # Loose Anchoring Signal Generation
    loose_core = data['Transition_Fractal_Momentum'] * data['Fractal_Transition_Efficiency']
    data['Loose_Signal'] = loose_core * data['Transition_Volume_Confirmation'] * data['Fractal_Acceleration']
    
    # Normal Anchoring Signal Generation
    normal_core = data['Fractal_Volume_Momentum'] * data['Volume_Fractal_Efficiency']
    data['Normal_Signal'] = normal_core * data['Opening_Anchor_Strength'] * data['Efficiency_Persistence']
    
    # Composite Fractal Factor
    data['Base_Fractal_Score'] = (data['Micro_Fractal_Momentum'] + data['Meso_Fractal_Momentum']) / 2
    data['Volume_Fractal_Score'] = data['Fractal_Volume_Momentum'] * data['Volume_Anchor_Persistence']
    
    # Regime-Adaptive Multiplier
    data['Regime_Adaptive_Multiplier'] = np.where(
        data['Tight_Anchoring_Regime'], 
        data['Tight_Signal'],
        np.where(
            data['Loose_Anchoring_Regime'],
            data['Loose_Signal'],
            data['Normal_Signal']
        )
    )
    
    # Final Alpha
    data['alpha'] = data['Base_Fractal_Score'] * data['Volume_Fractal_Score'] * data['Regime_Adaptive_Multiplier']
    
    return data['alpha']
