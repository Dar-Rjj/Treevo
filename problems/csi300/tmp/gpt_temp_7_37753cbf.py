import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Helper function to calculate fractal dimension proxy
    def calculate_fractal(series, window=5):
        return (series.rolling(window=window).max() - series.rolling(window=window).min()) / (series.rolling(window=window).std() + 1e-8)
    
    # Calculate fractal components
    df['price_fractal'] = calculate_fractal(df['close'])
    df['volume_fractal'] = calculate_fractal(df['volume'])
    df['amount_fractal'] = calculate_fractal(df['amount'])
    
    # Fractal-Momentum Structure
    df['fractal_weighted_acceleration'] = (
        (df['close'] - df['close'].shift(3)) / 
        (df['high'].shift(3) - df['low'].shift(3) + 1e-8) * 
        df['price_fractal'] * 
        (df['high'] - df['low']) / (df['high'].shift(2) - df['low'].shift(2) + 1e-8)
    )
    
    # Volume-Fractal Persistence
    volume_comparison = pd.DataFrame({
        't_minus_1': df['volume'] > df['volume'].shift(1),
        't_minus_2': df['volume'] > df['volume'].shift(2),
        't_minus_3': df['volume'] > df['volume'].shift(3),
        't_minus_4': df['volume'] > df['volume'].shift(4)
    })
    df['volume_fractal_persistence'] = (
        volume_comparison.sum(axis=1) / 4 * 
        (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
        df['volume_fractal']
    )
    
    # Fractal-Elastic Momentum
    df['fractal_elastic_momentum'] = (
        ((df['close'] / df['close'].shift(2) - 1) - (df['close'].shift(2) / df['close'].shift(4) - 1)) * 
        (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
        df['price_fractal']
    )
    
    # Fractal-Flow Asymmetry
    df['fractal_adaptive_flow'] = (
        (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * 
        (df['amount'] / df['amount'].shift(1) - 1) * 
        df['price_fractal']
    )
    
    # Amount-Fractal Compression
    range_condition = (df['high'] - df['low']) < (df['high'].shift(1) - df['low'].shift(1))
    compression_numerator = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) - (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['amount_fractal_compression'] = (
        compression_numerator * 
        (df['amount'] / df['amount'].shift(1) - 1) * 
        df['amount_fractal']
    )
    df.loc[~range_condition, 'amount_fractal_compression'] = 0
    
    # Fractal-Flow Divergence
    df['fractal_flow_divergence'] = df['fractal_adaptive_flow'] - df['amount_fractal_compression']
    
    # Fractal-Amount Regime Dynamics
    df['volume_fractal_amount_momentum'] = (
        ((df['amount'] / df['amount'].shift(1) - 1) - (df['amount'].shift(1) / df['amount'].shift(2) - 1)) * 
        (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
        df['amount_fractal']
    )
    
    df['fractal_amount_price_divergence'] = (
        df['volume_fractal_amount_momentum'] - 
        df['fractal_elastic_momentum'] * np.sign(df['fractal_elastic_momentum'])
    )
    
    df['fractal_amount_convergence'] = df['volume_fractal_amount_momentum'] * df['fractal_elastic_momentum']
    
    # Fractal-Gap Dynamics
    df['fractal_gap_absorption'] = (
        (df['close'] - df['open']) / (df['open'] - df['close'].shift(1) + 1e-8) * 
        (df['high'] - df['low']) / (df['high'].shift(2) - df['low'].shift(2) + 1e-8) * 
        df['price_fractal']
    )
    
    df['fractal_range_momentum'] = (
        ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) - 
         (df['high'].shift(2) - df['low'].shift(2)) / (df['high'].shift(3) - df['low'].shift(3) + 1e-8)) * 
        (df['amount'] / df['amount'].shift(1) - 1) * 
        df['amount_fractal']
    )
    
    df['fractal_gap_alignment'] = df['fractal_gap_absorption'] * df['fractal_range_momentum']
    
    # Regime-Adaptive Fractal Synthesis
    df['high_fractal_volume'] = df['fractal_weighted_acceleration'] * df['fractal_flow_divergence'] * df['fractal_amount_convergence']
    df['low_fractal_volume'] = df['fractal_elastic_momentum'] * df['fractal_amount_price_divergence'] * df['fractal_gap_alignment']
    df['transition_regime'] = df['volume_fractal_persistence'] * df['fractal_adaptive_flow'] * df['fractal_range_momentum']
    
    # Adaptive regime selection based on volume fractal
    volume_fractal_quantile = df['volume_fractal'].rolling(window=20).apply(lambda x: pd.Series(x).quantile(0.7), raw=True)
    df['regime_adaptive_fractal_synthesis'] = np.where(
        df['volume_fractal'] > volume_fractal_quantile,
        df['high_fractal_volume'],
        np.where(
            df['volume_fractal'] < volume_fractal_quantile.shift(5),
            df['low_fractal_volume'],
            df['transition_regime']
        )
    )
    
    # Adaptive Alpha Synthesis
    df['momentum_efficiency_component'] = df['fractal_elastic_momentum'] * df['fractal_flow_divergence']
    df['amount_regime_component'] = df['volume_fractal_amount_momentum'] * df['fractal_amount_price_divergence']
    
    # Final Alpha
    alpha = (
        df['regime_adaptive_fractal_synthesis'] * 
        df['momentum_efficiency_component'] * 
        df['amount_regime_component']
    )
    
    return alpha
