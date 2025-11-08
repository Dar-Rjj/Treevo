import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    
    This factor combines momentum across multiple timeframes, normalizes by volatility,
    incorporates volume divergence analysis, and adapts weights based on market regime.
    """
    
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['momentum_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['momentum_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Calculate returns for volatility estimation
    df['returns_3d'] = df['close'].pct_change(3)
    df['returns_10d'] = df['close'].pct_change(10)
    df['returns_20d'] = df['close'].pct_change(20)
    
    # Volatility Normalization
    df['vol_3d'] = df['returns_3d'].rolling(window=10, min_periods=5).std()
    df['vol_10d'] = df['returns_10d'].rolling(window=20, min_periods=10).std()
    df['vol_20d'] = df['returns_20d'].rolling(window=30, min_periods=15).std()
    
    # Normalize momentums by volatility
    df['norm_momentum_3d'] = df['momentum_3d'] / df['vol_3d']
    df['norm_momentum_10d'] = df['momentum_10d'] / df['vol_10d']
    df['norm_momentum_20d'] = df['momentum_20d'] / df['vol_20d']
    
    # Replace infinite values and handle NaN
    for col in ['norm_momentum_3d', 'norm_momentum_10d', 'norm_momentum_20d']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Volume Divergence Analysis
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10, min_periods=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=10).mean()
    
    df['volume_trend_short'] = df['volume'] / df['volume_ma_5']
    df['volume_trend_medium'] = df['volume'] / df['volume_ma_10']
    df['volume_trend_long'] = df['volume'] / df['volume_ma_20']
    
    # Calculate volume divergence score
    df['volume_divergence'] = 0.0
    
    # Positive momentum with declining volume → weak signal (negative divergence)
    mask_weak = (df['momentum_3d'] > 0) & (df['volume_trend_short'] < 0.9)
    df.loc[mask_weak, 'volume_divergence'] -= 0.5
    
    # Negative momentum with increasing volume → strong reversal potential (positive divergence)
    mask_reversal = (df['momentum_3d'] < 0) & (df['volume_trend_short'] > 1.1)
    df.loc[mask_reversal, 'volume_divergence'] += 1.0
    
    # Aligned momentum and volume → strong trend confirmation
    mask_aligned_up = (df['momentum_3d'] > 0) & (df['volume_trend_short'] > 1.05)
    mask_aligned_down = (df['momentum_3d'] < 0) & (df['volume_trend_short'] < 0.95)
    df.loc[mask_aligned_up, 'volume_divergence'] += 0.3
    df.loc[mask_aligned_down, 'volume_divergence'] -= 0.3
    
    # Regime-Based Weighting
    current_vol = df['returns_3d'].abs().rolling(window=10, min_periods=5).mean()
    vol_percentile = current_vol.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 60)) if len(x.dropna()) > 0 else np.nan
    )
    
    # Initialize regime weights
    df['regime_weight_short'] = 0.33
    df['regime_weight_medium'] = 0.33
    df['regime_weight_long'] = 0.34
    
    # High volatility regime: emphasize short-term signals
    high_vol_mask = vol_percentile > 0.6
    df.loc[high_vol_mask, 'regime_weight_short'] = 0.5
    df.loc[high_vol_mask, 'regime_weight_medium'] = 0.3
    df.loc[high_vol_mask, 'regime_weight_long'] = 0.2
    
    # Low volatility regime: emphasize long-term signals
    low_vol_mask = vol_percentile < 0.4
    df.loc[low_vol_mask, 'regime_weight_short'] = 0.2
    df.loc[low_vol_mask, 'regime_weight_medium'] = 0.3
    df.loc[low_vol_mask, 'regime_weight_long'] = 0.5
    
    # Final Alpha Factor Construction
    df['weighted_momentum'] = (
        df['norm_momentum_3d'] * df['regime_weight_short'] +
        df['norm_momentum_10d'] * df['regime_weight_medium'] +
        df['norm_momentum_20d'] * df['regime_weight_long']
    )
    
    # Apply volume divergence multiplier (1 + divergence score)
    df['alpha_factor'] = df['weighted_momentum'] * (1 + df['volume_divergence'])
    
    # Clean up and return
    alpha_series = df['alpha_factor'].copy()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
