import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum-Volume Convergence factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for different periods
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_21d'] = df['close'].pct_change(21)
    
    # Exponential Decay Momentum (λ = 0.94)
    lambda_val = 0.94
    decay_weights = {
        1: lambda_val ** 1,
        5: lambda_val ** 5,
        21: lambda_val ** 21
    }
    
    df['decay_momentum'] = (
        df['ret_1d'].fillna(0) * decay_weights[1] +
        df['ret_5d'].fillna(0) * decay_weights[5] +
        df['ret_21d'].fillna(0) * decay_weights[21]
    )
    
    # Momentum Acceleration
    df['momentum_accel'] = df['ret_5d'] - df['ret_10d']
    
    # Momentum trend consistency
    df['momentum_consistency'] = (
        (df['ret_1d'] > 0).astype(int) + 
        (df['ret_5d'] > 0).astype(int) + 
        (df['ret_21d'] > 0).astype(int)
    )
    
    # Volume Dynamics
    df['volume_5d'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_20d'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = np.log(df['volume_5d'] / df['volume_20d'])
    df['volume_accel'] = df['volume_ratio'].diff()
    df['volume_accel2'] = df['volume_accel'].diff()
    
    # Volume-Price Range Interaction
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['range_volume'] = np.log((df['high'] - df['low']) * df['volume'])
    
    # Volatility Measures
    df['vol_20d'] = df['ret_1d'].rolling(window=20, min_periods=1).std()
    df['atr_14d'] = df['true_range'].rolling(window=14, min_periods=1).mean()
    
    # Volatility Regime Classification
    df['vol_median_60d'] = df['vol_20d'].rolling(window=60, min_periods=1).median()
    df['vol_regime'] = np.where(df['vol_20d'] > df['vol_median_60d'], 'high', 'low')
    df['vol_trend'] = df['vol_20d'].diff(5) > 0
    
    # Volatility-Weighted Momentum
    df['vol_weighted_momentum'] = df['decay_momentum'] / (df['atr_14d'] + 1e-8)
    
    # Volume-Momentum Convergence Detection
    df['momentum_volume_alignment'] = (
        (df['ret_1d'] > 0) == (df['volume_accel'] > 0)
    ).astype(int)
    
    # Composite Alpha Generation
    for i in range(len(df)):
        if i < 60:  # Ensure sufficient data for calculations
            alpha.iloc[i] = 0
            continue
            
        row = df.iloc[i]
        
        # Core: Volatility-normalized decay momentum
        core_signal = row['vol_weighted_momentum']
        
        # Multiplier: Volume acceleration factor
        volume_multiplier = 1 + (row['volume_accel'] * 2)
        
        # Enhancer: Convergence/divergence detection
        convergence_enhancer = 1.0
        if row['momentum_volume_alignment'] == 1:
            convergence_enhancer = 1.2  # Boost for alignment
        else:
            convergence_enhancer = 0.8  # Reduce for divergence
        
        # Regime-specific weighting
        regime_weight = 1.0
        if row['vol_regime'] == 'high':
            # High volatility: emphasize momentum continuation
            regime_weight = 1.3 if row['momentum_accel'] * row['ret_1d'] > 0 else 0.7
        else:
            # Low volatility: mean reversion emphasis
            regime_weight = 0.8 if abs(row['ret_1d']) > 0.02 else 1.2
        
        # Combine components
        alpha_value = (
            core_signal * 
            volume_multiplier * 
            convergence_enhancer * 
            regime_weight
        )
        
        # Apply momentum consistency filter
        if row['momentum_consistency'] >= 2:
            alpha_value *= 1.1
        elif row['momentum_consistency'] <= 1:
            alpha_value *= 0.9
        
        alpha.iloc[i] = alpha_value
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
