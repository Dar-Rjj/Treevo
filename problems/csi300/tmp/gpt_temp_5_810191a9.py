import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Volatility Structure
    df['Micro_Volatility_Asymmetry'] = (df['high'] - df['open']) / (df['open'] - df['low'] + 1e-8)
    
    df['Meso_Volatility_Fractal'] = df['high'].rolling(5).apply(lambda x: x.max() - x.min()) / df['high'].rolling(10).apply(lambda x: x.max() - x.min())
    
    df['Macro_Volatility_Persistence'] = (df['high'] - df['low']) / (df['high'].shift(2) - df['low'].shift(2) + 1e-8)
    
    close_open_abs = np.abs(df['close'] - df['open']) + 1e-8
    hl_ratio = (df['high'] - df['low']) / close_open_abs
    df['Volatility_Entropy_Wave'] = hl_ratio * np.log(hl_ratio + 1e-8)
    
    # Quantum Momentum Pressure System
    hl_range_sq = (df['high'] - df['low']) ** 2 + 1e-8
    df['Bull_Pressure_Field'] = (df['close'] - df['low']) * (df['high'] - df['open']) / hl_range_sq
    df['Bear_Pressure_Field'] = (df['high'] - df['close']) * (df['open'] - df['low']) / hl_range_sq
    df['Pressure_Field_Gradient'] = df['Bull_Pressure_Field'] - df['Bear_Pressure_Field']
    
    bull_mask = df['close'] > df['open']
    bear_mask = df['close'] < df['open']
    df['Bull_Momentum_Intensity'] = ((df['close'] - df['open']) * df['volume'] * bull_mask).rolling(5).sum()
    df['Bear_Momentum_Intensity'] = ((df['open'] - df['close']) * df['volume'] * bear_mask).rolling(5).sum()
    
    momentum_ratio = df['Bull_Momentum_Intensity'] / (df['Bear_Momentum_Intensity'] + 1e-8)
    df['Quantum_Momentum_Asymmetry'] = df['Pressure_Field_Gradient'] * np.log(momentum_ratio + 1e-8)
    
    # Temporal Coherence Divergence
    df['Micro_Coherence'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    
    high_3 = df['high'].rolling(3).max()
    low_3 = df['low'].rolling(3).min()
    df['Meso_Coherence'] = (df['close'] - df['close'].shift(2)) / (high_3 - low_3 + 1e-8)
    
    high_5 = df['high'].rolling(5).max()
    low_5 = df['low'].rolling(5).min()
    df['Macro_Coherence'] = (df['close'] - df['close'].shift(5)) / (high_5 - low_5 + 1e-8)
    
    price_change = (df['close'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    volume_change = (df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 1e-8)
    df['Volume_Price_Divergence'] = price_change - volume_change
    
    df['Coherence_Phase_Alignment'] = np.sign(df['Micro_Coherence']) + np.sign(df['Meso_Coherence']) + np.sign(df['Macro_Coherence'])
    
    coherence_sum = df['Micro_Coherence'] + df['Meso_Coherence'] + df['Macro_Coherence']
    df['Temporal_Volume_Divergence'] = df['Volume_Price_Divergence'] * df['Coherence_Phase_Alignment'] * coherence_sum
    
    # Quantum Breakout Dynamics
    df['Range_Expansion_Signal'] = (df['high'] - df['low']) / (df['high'].shift(3) - df['low'].shift(3) + 1e-8)
    
    gap = np.abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    absorption = np.abs(df['close'] - df['open']) / (np.abs(df['open'] - df['close'].shift(1)) + 1e-8)
    direction = np.sign(df['close'] - df['open'])
    df['Gap_Absorption_Efficiency'] = gap * absorption * direction
    
    prev_mid = (df['high'].shift(1) + df['low'].shift(1)) / 2
    df['Breakout_Direction'] = np.sign(df['close'] - prev_mid)
    
    prev_range = df['high'].shift(1) - df['low'].shift(1) + 1e-8
    df['Breakout_Strength'] = np.abs(df['close'] - prev_mid) / prev_range
    
    volume_ratio = df['volume'] / (df['volume'].shift(1) + 1e-8)
    df['Quantum_Breakout_Amplifier'] = 1 + (df['Breakout_Direction'] * df['Breakout_Strength'] * df['Gap_Absorption_Efficiency'] * volume_ratio)
    
    # Multi-Scale Signal Construction
    df['Volatility_Momentum_Base'] = df['Micro_Volatility_Asymmetry'] * df['Quantum_Momentum_Asymmetry']
    df['Temporal_Enhanced_Signal'] = df['Volatility_Momentum_Base'] * df['Temporal_Volume_Divergence']
    df['Breakout_Enhanced_Signal'] = df['Temporal_Enhanced_Signal'] * df['Quantum_Breakout_Amplifier']
    df['Multi_Scale_Fusion'] = df['Breakout_Enhanced_Signal'] * df['Meso_Volatility_Fractal'] * df['Macro_Volatility_Persistence']
    
    # Quantum Memory Regimes
    df['Volatility_Memory_Echo'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    df['Volume_Memory_Echo'] = df['volume'] / (df['volume'].shift(1) + df['volume'].shift(2) + 1e-8)
    df['Price_Memory_Echo'] = np.abs(df['close'] - df['close'].shift(1)) / (np.abs(df['close'].shift(1) - df['close'].shift(2)) + 1e-8)
    df['Memory_Regime_Signal'] = df['Volatility_Memory_Echo'] * df['Volume_Memory_Echo'] * df['Price_Memory_Echo']
    
    # Regime Classification
    df['Volatility_Regime'] = 'Neutral'
    df.loc[df['Macro_Volatility_Persistence'] > 1.1, 'Volatility_Regime'] = 'Expanding'
    df.loc[df['Macro_Volatility_Persistence'] < 0.9, 'Volatility_Regime'] = 'Contracting'
    
    df['Momentum_Regime'] = 'Balanced'
    df.loc[df['Bull_Momentum_Intensity'] > 1.5 * df['Bear_Momentum_Intensity'], 'Momentum_Regime'] = 'Strong'
    df.loc[df['Bear_Momentum_Intensity'] > 1.5 * df['Bull_Momentum_Intensity'], 'Momentum_Regime'] = 'Weak'
    
    volume_3_sum = df['volume'].rolling(3).sum()
    df['Volume_Concentration'] = df['volume'] / (volume_3_sum + 1e-8)
    df['Volume_Regime'] = 'Normal'
    df.loc[df['Volume_Concentration'] > 0.4, 'Volume_Regime'] = 'High'
    df.loc[df['Volume_Concentration'] < 0.25, 'Volume_Regime'] = 'Low'
    
    # Regime-Adaptive Enhancement
    df['Expanding_Volatility_Alpha'] = df['Multi_Scale_Fusion'] * df['Volatility_Entropy_Wave'] * df['Quantum_Breakout_Amplifier']
    df['Contracting_Volatility_Alpha'] = df['Multi_Scale_Fusion'] * df['Memory_Regime_Signal'] * df['Volume_Memory_Echo']
    df['Neutral_Volatility_Alpha'] = df['Multi_Scale_Fusion'] * df['Temporal_Volume_Divergence'] * df['Meso_Volatility_Fractal']
    df['Transition_Alpha'] = (df['Expanding_Volatility_Alpha'] + df['Contracting_Volatility_Alpha'] + df['Neutral_Volatility_Alpha']) / 3
    
    # Quantum Stability Field
    typical_price = (df['open'] + df['high'] + df['low']) / 3
    df['Price_Stability'] = 1 - np.abs(df['close'] - typical_price) / (df['high'] - df['low'] + 1e-8)
    df['Volume_Stability'] = 1 - np.abs(df['volume'] - df['volume'].shift(1)) / (df['volume'] + df['volume'].shift(1) + 1e-8)
    df['Quantum_Stability_Multiplier'] = df['Price_Stability'] * df['Volume_Stability']
    
    # Final Alpha Synthesis
    conditions = [
        (df['Volatility_Regime'] == 'Expanding') & (df['Momentum_Regime'] == 'Strong'),
        (df['Volatility_Regime'] == 'Contracting') & (df['Volume_Regime'] == 'High'),
        (df['Volatility_Regime'] == 'Neutral') & (df['Momentum_Regime'] == 'Balanced')
    ]
    
    choices = [
        df['Expanding_Volatility_Alpha'],
        df['Contracting_Volatility_Alpha'],
        df['Neutral_Volatility_Alpha']
    ]
    
    df['Selected_Alpha'] = np.select(conditions, choices, default=df['Transition_Alpha'])
    df['Stability_Enhanced_Alpha'] = df['Selected_Alpha'] * df['Quantum_Stability_Multiplier']
    df['Multi_Scale_Volatility_Momentum_Resonance_Alpha'] = df['Stability_Enhanced_Alpha'] * (1 + np.abs(df['Temporal_Volume_Divergence']))
    
    return df['Multi_Scale_Volatility_Momentum_Resonance_Alpha']
