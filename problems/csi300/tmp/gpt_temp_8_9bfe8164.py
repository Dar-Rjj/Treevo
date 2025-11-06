import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate basic price differences and ranges
    df['close_diff_1'] = df['close'].diff(1)
    df['close_diff_2'] = df['close'].diff(2)
    df['price_range'] = df['high'] - df['low']
    df['price_range_prev'] = df['price_range'].shift(1)
    df['price_range_prev2'] = df['price_range'].shift(2)
    df['price_range_prev3'] = df['price_range'].shift(3)
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['close_open_diff'] = df['close'] - df['open']
    df['open_close_prev_diff'] = df['open'] - df['close'].shift(1)
    df['price_mid'] = (df['high'] + df['low']) / 2
    
    # Calculate price fractal dimension (simplified as rolling price complexity)
    window = 5
    df['price_volatility'] = df['close'].rolling(window=window).std()
    df['price_range_avg'] = df['price_range'].rolling(window=window).mean()
    df['price_fractal'] = df['price_volatility'] / (df['price_range_avg'] + 1e-8)
    
    # Calculate price entropy (information content in price movements)
    df['price_change'] = df['close'].pct_change()
    df['price_entropy'] = -df['price_change'].rolling(window=window).apply(
        lambda x: np.sum(x * np.log(np.abs(x) + 1e-8)) if len(x) == window else np.nan
    )
    
    # Calculate volume entropy
    df['volume_change'] = df['volume'].pct_change()
    df['volume_entropy'] = -df['volume_change'].rolling(window=window).apply(
        lambda x: np.sum(x * np.log(np.abs(x) + 1e-8)) if len(x) == window else np.nan
    )
    
    # Fractal Entropy Momentum Asymmetry components
    df['entropy_momentum_acceleration'] = (
        (df['close_diff_2'] / (df['price_range_prev2'] + 1e-8)) * 
        df['price_fractal'] * 
        (df['close_diff_1'] / (df['price_range'] + 1e-8)) * 
        df['price_entropy'] * 
        np.sign(df['close_diff_1'])
    )
    
    df['entropy_price_volume_alignment'] = (
        ((df['close'] / df['close'].shift(1) - 1) - (df['close'].shift(1) / df['close'].shift(2) - 1)) *
        (df['price_range'] / (df['price_range_prev'] + 1e-8)) *
        df['price_fractal'] *
        df['price_entropy'] *
        (df['volume_ratio'] - 1) *
        np.sign(df['close_diff_1'])
    )
    
    df['volume_weighted_entropy_flow'] = (
        df['entropy_price_volume_alignment'] *
        (df['close_open_diff'] / (df['price_range'] + 1e-8)) *
        np.sign(df['close_open_diff'])
    )
    
    # Fractal Entropy Volatility Asymmetry components
    df['intraday_entropy_flow'] = (
        (df['price_range'] / (df['open_close_prev_diff'] + 1e-8)) *
        df['volume'] *
        df['price_entropy'] *
        np.sign(df['close_open_diff'])
    )
    
    df['entropy_volatility_momentum'] = (
        (df['price_range'] / (df['price_range_prev'] + 1e-8) - 
         df['price_range_prev2'] / (df['price_range_prev3'] + 1e-8)) *
        df['price_entropy'] *
        np.sign(df['close_diff_1'])
    )
    
    df['entropy_gap_range_dynamics'] = (
        (df['open_close_prev_diff'] / (df['price_range'] + 1e-8)) *
        (df['price_range'] / (df['price_range_prev'] + 1e-8)) *
        df['price_fractal'] *
        df['price_entropy'] *
        np.sign(df['close_open_diff'])
    )
    
    # Asymmetric Entropy Patterns components
    df['directional_entropy_bias'] = (
        ((df['close'] - df['price_mid']) / (df['price_range'] + 1e-8)) *
        df['price_entropy'] *
        np.sign(df['close_open_diff']) *
        np.sign(df['volume'] - df['volume'].shift(1))
    )
    
    df['volume_entropy_asymmetry'] = (
        df['intraday_entropy_flow'] *
        df['volume_ratio'] *
        df['volume_entropy'] *
        np.sign(df['close_diff_1'])
    )
    
    df['session_entropy_divergence'] = (
        df['intraday_entropy_flow'] - 
        ((df['close_open_diff'] / (df['price_range'] + 1e-8)) * df['volume'] * df['price_entropy'])
    )
    
    # Multi-Scale Fractal Integration components
    df['price_volume_entropy_coherence'] = (
        df['entropy_price_volume_alignment'] * df['volume_weighted_entropy_flow']
    )
    
    df['entropy_flow_adaptation'] = (
        df['volume_weighted_entropy_flow'] * df['entropy_gap_range_dynamics']
    )
    
    df['fractal_entropy_divergence'] = (
        df['entropy_price_volume_alignment'] - 
        df['volume_weighted_entropy_flow'] * np.sign(df['volume_weighted_entropy_flow'])
    )
    
    # Multi-Scale Alpha Synthesis
    df['core_entropy_momentum'] = (
        df['entropy_momentum_acceleration'] * df['price_volume_entropy_coherence']
    )
    
    df['adaptive_entropy_flow'] = (
        df['entropy_flow_adaptation'] * df['directional_entropy_bias']
    )
    
    # Final alpha factor
    df['alpha_factor'] = (
        df['core_entropy_momentum'] * 
        df['adaptive_entropy_flow'] * 
        df['fractal_entropy_divergence']
    )
    
    return df['alpha_factor']
