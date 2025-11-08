import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Momentum-Volume Divergence with Cross-Sectional Ranking factor
    """
    df = data.copy()
    
    # Raw Momentum Calculation
    # Price Momentum Components
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Components
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Regime-Aware Smoothing
    # Amount-Based Regime Classification
    df['amount_acceleration'] = (df['amount'] / df['amount'].shift(5)) - (df['amount'].shift(5) / df['amount'].shift(10))
    
    # Define regime conditions
    high_participation = df['amount_acceleration'] > 0.1
    low_participation = df['amount_acceleration'] < -0.1
    normal_regime = ~high_participation & ~low_participation
    
    # Adaptive Exponential Smoothing function
    def adaptive_ema(series, regime_mask_high, regime_mask_low, regime_mask_normal):
        result = pd.Series(index=series.index, dtype=float)
        
        # High participation regime: alpha = 0.5
        high_data = series[regime_mask_high]
        if not high_data.empty:
            result[regime_mask_high] = high_data.ewm(alpha=0.5, adjust=False).mean()
        
        # Low participation regime: alpha = 0.2
        low_data = series[regime_mask_low]
        if not low_data.empty:
            result[regime_mask_low] = low_data.ewm(alpha=0.2, adjust=False).mean()
        
        # Normal regime: alpha = 0.3
        normal_data = series[regime_mask_normal]
        if not normal_data.empty:
            result[regime_mask_normal] = normal_data.ewm(alpha=0.3, adjust=False).mean()
        
        return result
    
    # Apply regime-specific EMA to all momentum components
    momentum_components = ['price_momentum_5d', 'price_momentum_10d', 'price_momentum_20d',
                         'volume_momentum_5d', 'volume_momentum_10d', 'volume_momentum_20d']
    
    for col in momentum_components:
        df[f'ema_{col}'] = adaptive_ema(df[col], high_participation, low_participation, normal_regime)
    
    # Momentum Acceleration Analysis
    # Price Momentum Acceleration
    df['price_accel_5d'] = df['ema_price_momentum_5d'] - df['ema_price_momentum_5d'].shift(1)
    df['price_accel_10d'] = df['ema_price_momentum_10d'] - df['ema_price_momentum_10d'].shift(1)
    df['price_accel_20d'] = df['ema_price_momentum_20d'] - df['ema_price_momentum_20d'].shift(1)
    
    # Volume Momentum Acceleration
    df['volume_accel_5d'] = df['ema_volume_momentum_5d'] - df['ema_volume_momentum_5d'].shift(1)
    df['volume_accel_10d'] = df['ema_volume_momentum_10d'] - df['ema_volume_momentum_10d'].shift(1)
    df['volume_accel_20d'] = df['ema_volume_momentum_20d'] - df['ema_volume_momentum_20d'].shift(1)
    
    # Cross-Sectional Ranking Implementation
    # Calculate cross-sectional percentiles for each date
    def calculate_cross_sectional_ranks(df, column):
        return df.groupby(df.index)[column].transform(lambda x: x.rank(pct=True))
    
    # Universe-Based Relative Ranking
    df['price_rank'] = calculate_cross_sectional_ranks(df, 'ema_price_momentum_10d')
    df['volume_rank'] = calculate_cross_sectional_ranks(df, 'ema_volume_momentum_10d')
    df['divergence_rank'] = df['price_rank'] - df['volume_rank']
    
    # Acceleration-Based Ranking
    df['price_accel_rank'] = calculate_cross_sectional_ranks(df, 'price_accel_10d')
    df['volume_accel_rank'] = calculate_cross_sectional_ranks(df, 'volume_accel_10d')
    df['accel_divergence_rank'] = df['price_accel_rank'] - df['volume_accel_rank']
    
    # Additional rankings for regime signals
    df['price_momentum_rank'] = calculate_cross_sectional_ranks(df, 'ema_price_momentum_10d')
    df['volume_momentum_rank'] = calculate_cross_sectional_ranks(df, 'ema_volume_momentum_10d')
    
    # Regime-Adaptive Signal Construction
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # High Participation Regime Signals
    high_mask = high_participation
    if high_mask.any():
        high_signal = (
            0.7 * df.loc[high_mask, 'divergence_rank'] +
            0.2 * df.loc[high_mask, 'volume_accel_rank'] +
            0.1 * df.loc[high_mask, 'price_momentum_rank']
        )
        factor_values[high_mask] = high_signal
    
    # Low Participation Regime Signals
    low_mask = low_participation
    if low_mask.any():
        low_signal = (
            0.6 * df.loc[low_mask, 'price_accel_rank'] +
            0.3 * df.loc[low_mask, 'divergence_rank'] +
            0.1 * df.loc[low_mask, 'volume_momentum_rank']
        )
        factor_values[low_mask] = low_signal
    
    # Normal Regime Signals
    normal_mask = normal_regime
    if normal_mask.any():
        normal_signal = (
            0.5 * df.loc[normal_mask, 'divergence_rank'] +
            0.3 * df.loc[normal_mask, 'price_accel_rank'] +
            0.2 * df.loc[normal_mask, 'volume_accel_rank']
        )
        factor_values[normal_mask] = normal_signal
    
    return factor_values
