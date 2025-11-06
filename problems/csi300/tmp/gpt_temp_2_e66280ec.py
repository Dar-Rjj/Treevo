import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Aware Multi-Timeframe Momentum Factor
    Combines price acceleration, volume confirmation, and volatility-adaptive scaling
    with dynamic regime adjustments for robust factor generation.
    """
    df = data.copy()
    
    # Multi-Timeframe Price Acceleration
    # Recent Price Acceleration
    df['mom_1d'] = df['close'] / df['close'].shift(1) - 1
    df['mom_3d'] = df['close'] / df['close'].shift(3) - 1
    df['accel_ratio'] = df['mom_1d'] / (df['mom_3d'] + 1e-8)
    
    # Medium-term Momentum Stability
    df['mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['mom_10d'] = df['close'] / df['close'].shift(10) - 1
    df['mom_persistence'] = np.sign(df['mom_5d']) * np.sign(df['mom_10d']) * np.minimum(np.abs(df['mom_5d']), np.abs(df['mom_10d']))
    
    # Timeframe Alignment Signal
    df['alignment_strength'] = (np.sign(df['mom_1d']) * np.sign(df['mom_5d']) * 
                               (np.abs(df['mom_1d']) + np.abs(df['mom_5d'])) / 2)
    
    # Volume-Confirmed Momentum
    # Volume Acceleration Profile
    df['vol_1d'] = df['volume'] / df['volume'].shift(1) - 1
    df['vol_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['vol_accel'] = df['vol_1d'] / (df['vol_5d'] + 1e-8)
    
    # Momentum-Volume Integration
    df['vol_weighted_mom'] = df['mom_5d'] * df['vol_accel']
    df['vol_confirmation'] = (df['volume'] > df['volume'].shift(5)).astype(float)
    df['enhanced_signal'] = df['mom_5d'] * (1 + df['vol_confirmation'] * np.sign(df['mom_5d']))
    
    # Volatility-Adaptive Scaling
    # Dynamic Volatility Measure
    df['current_vol'] = (df['high'] - df['low']) / df['close']
    df['vol_5d_ago'] = (df['high'].shift(5) - df['low'].shift(5)) / df['close'].shift(5)
    df['vol_trend'] = df['current_vol'] / (df['vol_5d_ago'] + 1e-8)
    df['vol_accel_ind'] = df['vol_trend'] - 1
    
    # Risk-Adjusted Momentum
    df['vol_normalized_mom'] = df['mom_5d'] / (df['current_vol'] + 1e-8)
    df['vol_regime_mult'] = 1 / (1 + np.abs(df['vol_accel_ind']))
    
    # Regime-Adaptive Factor Composition
    # Market State Detection
    df['vol_20d_median'] = df['current_vol'].rolling(window=20).median()
    df['vol_regime'] = (df['current_vol'] > df['vol_20d_median']).astype(float)
    
    df['vol_20d_median_abs'] = df['volume'].rolling(window=20).median()
    df['volume_regime'] = (df['volume'] > df['vol_20d_median_abs']).astype(float)
    
    df['mom_10d_mag'] = df['close'].pct_change(10).abs()
    df['mom_consistency'] = df['mom_5d'].rolling(window=5).apply(lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan)
    df['trend_regime'] = ((df['mom_10d_mag'] > df['mom_10d_mag'].rolling(window=20).median()) & 
                         (df['mom_consistency'] > 0.6)).astype(float)
    
    # Dynamic Factor Adjustment
    df['vol_coef'] = 1 - 0.3 * df['vol_regime']  # High volatility: apply damping
    df['volume_coef'] = 0.7 + 0.3 * df['volume_regime']  # Low volume: reduce magnitude
    df['trend_coef'] = 1 + 0.2 * df['trend_regime']  # Strong trend: enhance persistence
    
    # Base factor combining momentum, volume, and volatility components
    df['base_factor'] = (df['alignment_strength'] * 0.4 + 
                        df['enhanced_signal'] * 0.3 + 
                        df['vol_normalized_mom'] * 0.3)
    
    # Final regime-adaptive factor
    df['factor'] = df['base_factor'] * df['vol_coef'] * df['volume_coef'] * df['trend_coef']
    
    # Apply smoothing for strong trend regimes
    df['factor_smoothed'] = df['factor'].copy()
    strong_trend_mask = df['trend_regime'] == 1
    df.loc[strong_trend_mask, 'factor_smoothed'] = df.loc[strong_trend_mask, 'factor'].rolling(window=3, min_periods=1).mean()
    
    return df['factor_smoothed']
