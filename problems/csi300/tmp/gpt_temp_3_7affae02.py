import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Regime Volume Acceleration Alpha
    
    This factor combines volatility regime classification with volume acceleration
    and price reversion signals to generate alpha predictions.
    """
    df = data.copy()
    
    # Volatility Regime Classification
    # Calculate daily volatility proxy using high-low range normalized by previous close
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Establish volatility baseline using 60-day rolling median
    df['vol_baseline'] = df['daily_range'].rolling(window=60, min_periods=30).median()
    
    # Classify volatility regimes
    conditions = [
        df['daily_range'] > (1.5 * df['vol_baseline']),  # High volatility
        df['daily_range'] < (0.8 * df['vol_baseline']),  # Low volatility
    ]
    choices = [2, 0]  # 2=High, 1=Normal, 0=Low
    df['vol_regime'] = np.select(conditions, choices, default=1)
    
    # Volume Acceleration Component
    # Calculate volume momentum using median-based approach
    vol_5d_median = df['volume'].rolling(window=5, min_periods=3).median()
    vol_20d_median = df['volume'].rolling(window=20, min_periods=10).median()
    df['vol_acceleration'] = (vol_5d_median / vol_20d_median) - 1
    
    # Adaptive scaling using median-based normalization
    vol_acc_magnitude = df['vol_acceleration'].abs().rolling(window=20, min_periods=10).median()
    df['scaled_vol_acc'] = df['vol_acceleration'] / (vol_acc_magnitude + 1e-8)
    
    # Price Reversion Signal
    # Calculate short-term price movement
    df['price_3d_return'] = df['close'] / df['close'].shift(3) - 1
    
    # Generate reversion expectation (negative of recent returns for mean reversion)
    df['reversion_signal'] = -df['price_3d_return']
    
    # Scale by absolute return magnitude with robust median adjustment
    return_magnitude = df['price_3d_return'].abs().rolling(window=10, min_periods=5).median()
    df['scaled_reversion'] = df['reversion_signal'] / (return_magnitude + 1e-8)
    
    # Multiplicative Signal Integration
    # Initialize base signal
    df['base_signal'] = df['scaled_reversion'] * (1 + df['scaled_vol_acc'])
    
    # Apply regime-specific multipliers
    conditions_multiplier = [
        (df['vol_regime'] == 2) & (df['vol_acceleration'] > 0.3),  # High vol, strong acc
        (df['vol_regime'] == 2) & (df['vol_acceleration'] > 0.1) & (df['vol_acceleration'] <= 0.3),  # High vol, moderate acc
        (df['vol_regime'] == 2),  # High vol, no acceleration
        (df['vol_regime'] == 0),  # Low volatility
    ]
    multiplier_choices = [2.0, 1.5, 1.2, 0.5]
    df['regime_multiplier'] = np.select(conditions_multiplier, multiplier_choices, default=1.0)
    
    # Final alpha factor
    df['alpha'] = df['base_signal'] * df['regime_multiplier']
    
    # Apply final robust scaling using rolling median absolute deviation
    alpha_mad = df['alpha'].rolling(window=20, min_periods=10).apply(lambda x: np.median(np.abs(x - np.median(x))))
    df['final_alpha'] = df['alpha'] / (alpha_mad + 1e-8)
    
    return df['final_alpha']
