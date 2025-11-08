import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Fractal Association
    # Fractal dimension estimation
    df['range_5d_ratio'] = (df['high'] - df['low']) / (df['high'].shift(5) - df['low'].shift(5))
    df['range_10d_ratio'] = (df['high'] - df['low']) / (df['high'].shift(10) - df['low'].shift(10))
    df['fractal_consistency'] = df['range_5d_ratio'].rolling(window=5).std() / df['range_10d_ratio'].rolling(window=10).std()
    
    # Volume clustering patterns
    df['volume_burst'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_persistence'] = df['volume_burst'].rolling(window=3).apply(lambda x: np.sum(np.sign(x) == np.sign(x[0])) if len(x) == 3 else np.nan)
    
    # Microstructure Entropy Dynamics
    df['intraday_complexity'] = (df['high'] - df['low']) / (np.abs(df['close'] - df['open']) + 1e-8)
    df['overnight_absorption'] = np.abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    df['price_jitter'] = (np.abs(df['high'] - df['open']) + np.abs(df['low'] - df['open']) + np.abs(df['close'] - df['open'])) / (df['high'] - df['low'] + 1e-8)
    
    # Cross-Timeframe Momentum Resonance
    df['ultra_short_resonance'] = (df['close'] - df['close'].shift(1)) * (df['close'].shift(1) - df['close'].shift(2))
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['medium_resonance'] = np.sign(df['momentum_5d']) * np.sign(df['momentum_10d'])
    
    # Liquidity Gradient & Flow Asymmetry
    df['directional_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['volume_acceleration'] = df['volume'] / df['volume'].rolling(window=5).mean() - 1
    df['flow_asymmetry'] = df['directional_pressure'] * df['volume_acceleration']
    
    # Adaptive Entropy-Momentum Synthesis
    # Fractal-momentum coupling
    df['fractal_momentum_coupling'] = df['fractal_consistency'].rolling(window=5).mean() * df['momentum_5d']
    
    # Entropy-weighted momentum
    entropy_measure = (df['intraday_complexity'] + df['overnight_absorption'] + df['price_jitter']) / 3
    df['entropy_weighted_momentum'] = df['momentum_5d'] / (entropy_measure + 1e-8)
    
    # Cross-resonance validation
    df['resonance_alignment'] = df['ultra_short_resonance'] * df['medium_resonance'] * np.sign(df['momentum_5d'])
    
    # Liquidity-confirmed momentum
    df['liquidity_confirmation'] = df['momentum_5d'] * df['flow_asymmetry'] * (1 + df['volume_acceleration'])
    
    # Final alpha factor synthesis
    alpha = (df['fractal_momentum_coupling'] * 0.2 + 
             df['entropy_weighted_momentum'] * 0.3 + 
             df['resonance_alignment'] * 0.25 + 
             df['liquidity_confirmation'] * 0.25)
    
    return alpha
