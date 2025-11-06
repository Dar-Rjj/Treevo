import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Fractal Quantum State Detection
    # Intraday-Fractal Entanglement
    df['Fractal_Momentum_Component'] = (df['close'] - df['close'].shift(2)) / (df['high'].shift(2) - df['low'].shift(2))
    df['Intraday_Reversal_Component'] = (df['close'] - df['open']) / df['open'] * (-1)
    df['Fractal_Intraday_Phase_Coherence'] = np.sign(df['Fractal_Momentum_Component']) * np.sign(df['Intraday_Reversal_Component'])
    
    # Volume-Volatility Quantum State
    df['Volume_Concentration'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['open']
    df['Volume_Volatility_Entanglement'] = df['Volume_Concentration'] * df['Intraday_Volatility']
    
    # Quantum Fractal Regime Classification
    df['Fractal_Compression'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    df['Volatility_Ratio'] = df['close'].rolling(window=5).std() / df['close'].rolling(window=20).std()
    df['Quantum_State_Purity'] = abs(df['close'] - df['open']).rolling(window=3).apply(lambda x: np.sum(np.sign(x))) / 3
    
    # Entangled Momentum Framework
    # Quantum Fractal Tunneling
    df['Fractal_Resistance_Breakthrough'] = (df['high'] - df['high'].rolling(window=3).apply(lambda x: x.iloc[:-1].max())) / (df['high'] - df['low'])
    df['Fractal_Support_Penetration'] = (df['low'].rolling(window=3).apply(lambda x: x.iloc[:-1].min()) - df['low']) / (df['high'] - df['low'])
    df['Tunneling_Strength'] = np.where((df['Fractal_Resistance_Breakthrough'] > 0) | (df['Fractal_Support_Penetration'] > 0), df['Volume_Concentration'], 0)
    
    # Fractal Quantum Interference
    close_diff = df['close'] - df['close'].shift(1)
    df['Constructive_Fractal_Interference'] = (close_diff * np.sign(df['Fractal_Momentum_Component'])).rolling(window=3).sum()
    df['Destructive_Fractal_Interference'] = (close_diff * -np.sign(df['Fractal_Momentum_Component'])).rolling(window=3).sum()
    df['Net_Fractal_Interference'] = abs(df['Constructive_Fractal_Interference']) - abs(df['Destructive_Fractal_Interference'])
    
    # Quantum Fractal Decoherence
    df['Fractal_Price_Decoherence'] = df['Fractal_Momentum_Component'].rolling(window=3).std() / (df['close'] / df['close'].shift(1) - 1).rolling(window=3).std()
    df['Intraday_Fractal_Decoherence'] = df['Intraday_Reversal_Component'].rolling(window=3).std() / df['Fractal_Momentum_Component'].rolling(window=3).std()
    df['Quantum_Decoherence_Strength'] = 1 / (1 + abs(df['Fractal_Price_Decoherence']))
    
    # Adaptive Quantum Fractal Construction
    # Regime-Adaptive Signal Generation
    regime_factor = pd.Series(index=df.index, dtype=float)
    
    # High Volatility Quantum
    high_vol_mask = df['Volatility_Ratio'] > 1.2
    core_quantum_signal = df['Fractal_Momentum_Component'] * df['Volume_Volatility_Entanglement']
    regime_factor[high_vol_mask] = core_quantum_signal[high_vol_mask] / df['Quantum_Decoherence_Strength'][high_vol_mask]
    
    # Low Volatility Quantum
    low_vol_mask = df['Volatility_Ratio'] < 0.8
    volume_adjusted_fractal = df['Fractal_Momentum_Component'] * df['Volume_Concentration']
    regime_factor[low_vol_mask] = volume_adjusted_fractal[low_vol_mask] * df['Fractal_Compression'][low_vol_mask]
    
    # Normal Volatility Quantum
    normal_vol_mask = (df['Volatility_Ratio'] >= 0.8) & (df['Volatility_Ratio'] <= 1.2)
    base_quantum_signal = df['Fractal_Momentum_Component'] * df['Net_Fractal_Interference']
    regime_factor[normal_vol_mask] = base_quantum_signal[normal_vol_mask] / df['Quantum_Decoherence_Strength'][normal_vol_mask]
    
    # Quantum State Filtering
    filtered_factor = pd.Series(index=df.index, dtype=float)
    
    # Pure State Enhancement
    pure_mask = df['Quantum_State_Purity'] > 0.8
    filtered_factor[pure_mask] = regime_factor[pure_mask] * 1.2
    
    # Mixed State Reduction
    mixed_mask = df['Quantum_State_Purity'] < 0.3
    filtered_factor[mixed_mask] = regime_factor[mixed_mask] * 0.7
    
    # Transition State Normal
    transition_mask = (df['Quantum_State_Purity'] >= 0.3) & (df['Quantum_State_Purity'] <= 0.8)
    filtered_factor[transition_mask] = regime_factor[transition_mask]
    
    # Tunneling Amplification
    breakthrough_detection = (df['Fractal_Resistance_Breakthrough'] > 0.1) | (df['Fractal_Support_Penetration'] > 0.1)
    tunneling_momentum = df['Tunneling_Strength'] * df['Fractal_Momentum_Component']
    
    # Signal Integration
    signal_integration = pd.Series(index=df.index, dtype=float)
    signal_integration[breakthrough_detection] = filtered_factor[breakthrough_detection] * (1 + tunneling_momentum[breakthrough_detection])
    signal_integration[~breakthrough_detection] = filtered_factor[~breakthrough_detection]
    
    # Quantum Fractal Alpha Factor
    divergence_confirmation = df['Fractal_Intraday_Phase_Coherence'] < 0
    
    final_alpha = pd.Series(index=df.index, dtype=float)
    final_alpha[divergence_confirmation] = signal_integration[divergence_confirmation] * (1 + 0.5 * abs(df['Net_Fractal_Interference'][divergence_confirmation]))
    final_alpha[~divergence_confirmation] = signal_integration[~divergence_confirmation]
    
    return final_alpha
