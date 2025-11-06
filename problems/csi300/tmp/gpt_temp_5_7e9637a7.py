import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum-Decay Asymmetric Reversal with Volume-Weighted Regime Switching
    """
    df = data.copy()
    
    # Calculate basic price features
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_gap'] = (df['close'] - df['open']) / df['open']
    
    # Momentum-Decay Asymmetric Reversal Components
    # Short-term momentum features
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_accel'] = df['momentum_5'] - df['momentum_10'].shift(5)
    
    # Acceleration-Deceleration Divergence
    df['price_acceleration'] = df['momentum_5'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 5 else np.nan, raw=False
    )
    df['price_deceleration'] = -df['price_acceleration'].shift(5)
    df['accel_decel_divergence'] = df['price_acceleration'] - df['price_deceleration']
    
    # Momentum Persistence Fracture Points
    df['momentum_trend'] = df['momentum_5'].rolling(window=10).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] * x.iloc[i-1] > 0]) / len(x) 
        if len(x) == 10 else np.nan, raw=False
    )
    df['momentum_breakdown'] = (df['momentum_trend'] < 0.6).astype(int)
    
    # Return Asymmetry During Momentum Phases
    df['positive_momentum_skew'] = df['returns'].rolling(window=10).apply(
        lambda x: x[x > 0].skew() if len(x[x > 0]) > 2 else 0, raw=False
    )
    df['negative_momentum_skew'] = df['returns'].rolling(window=10).apply(
        lambda x: x[x < 0].skew() if len(x[x < 0]) > 2 else 0, raw=False
    )
    df['momentum_asymmetry'] = df['positive_momentum_skew'] - df['negative_momentum_skew']
    
    # Reversal Strength Measurement
    # Oversold/Overbought Rebound Intensity
    df['rsi_14'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / 
                                    df['close'].diff().clip(upper=0).abs().rolling(14).mean())))
    df['oversold_rebound'] = ((df['rsi_14'] < 30) & (df['returns'] > 0)).astype(int) * df['returns'].abs()
    df['overbought_reversal'] = ((df['rsi_14'] > 70) & (df['returns'] < 0)).astype(int) * df['returns'].abs()
    
    # Support/Resistance Break Failure Rates
    df['resistance_level'] = df['high'].rolling(window=20).max()
    df['support_level'] = df['low'].rolling(window=20).min()
    df['breakout_failure'] = ((df['close'] > df['resistance_level'].shift(1)) & 
                             (df['close'] < df['resistance_level'])).astype(int)
    df['breakdown_failure'] = ((df['close'] < df['support_level'].shift(1)) & 
                              (df['close'] > df['support_level'])).astype(int)
    
    # Decay-Rate Asymmetry
    # Fast Decay vs Slow Recovery Dynamics
    df['momentum_decay_rate'] = df['momentum_5'].rolling(window=5).apply(
        lambda x: (x.iloc[0] - x.iloc[-1]) / 5 if len(x) == 5 else np.nan, raw=False
    )
    df['recovery_rate'] = df['returns'].rolling(window=5).apply(
        lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0, raw=False
    )
    df['decay_recovery_ratio'] = df['momentum_decay_rate'].abs() / (df['recovery_rate'] + 1e-6)
    
    # Volume-Weighted Decay Acceleration
    df['volume_weighted_decay'] = df['momentum_decay_rate'] * (df['volume'] / df['volume'].rolling(20).mean())
    
    # Volume-Weighted Regime Switching Components
    # Volume Concentration Regime Detection
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_std_20'] = df['volume'].rolling(20).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / (df['volume_std_20'] + 1e-6)
    
    # High Volume Cluster Persistence
    df['high_volume_cluster'] = (df['volume_zscore'] > 1).astype(int)
    df['volume_cluster_persistence'] = df['high_volume_cluster'].rolling(window=5).sum()
    
    # Volume Spike Regime Classification
    df['volume_spike_regime'] = pd.cut(df['volume_zscore'], 
                                      bins=[-np.inf, -1, 1, np.inf], 
                                      labels=[0, 1, 2]).astype(float)
    
    # Price-Volume Regime Coupling
    df['price_volume_correlation'] = df['returns'].rolling(window=10).corr(df['volume'])
    df['volume_impact_multiplier'] = df['returns'].abs() * df['volume_zscore'].abs()
    
    # Multi-Timeframe Regime Synchronization
    df['intraday_regime_consistency'] = df['volume_spike_regime'].rolling(window=3).apply(
        lambda x: 1 if len(set(x)) == 1 else 0, raw=False
    )
    
    # Asymmetric Signal Integration Framework
    # Momentum-Decay to Regime Transition Mapping
    df['decay_regime_precursor'] = df['momentum_decay_rate'] * df['volume_zscore']
    
    # Regime-Dependent Reversal Strength Calibration
    df['regime_adjusted_reversal'] = df['oversold_rebound'] * (1 + df['volume_zscore'].clip(0, 2))
    
    # Volume-Weighted Signal Amplification
    df['high_volume_signal_enhancement'] = df['accel_decel_divergence'] * (1 + df['volume_zscore'].clip(0, 1))
    df['low_volume_signal_suppression'] = df['momentum_asymmetry'] * (1 - (df['volume_zscore'].clip(-1, 0)).abs())
    
    # Dynamic Alpha Construction
    # Asymmetric Interaction Terms
    df['momentum_decay_volume_intensity'] = df['volume_weighted_decay'] * df['volume_cluster_persistence']
    df['reversal_regime_probability'] = df['regime_adjusted_reversal'] * df['intraday_regime_consistency']
    df['decay_asymmetry_volume_concentration'] = df['decay_recovery_ratio'] * df['volume_zscore'].abs()
    
    # Regime-Adaptive Signal Combination
    high_volume_mask = df['volume_zscore'] > 1
    low_volume_mask = df['volume_zscore'] < -0.5
    transition_mask = (df['volume_zscore'].abs().rolling(3).std() > 0.8)
    
    # High Volume Concentration Signals
    high_volume_signals = (df['momentum_decay_volume_intensity'] * 0.4 + 
                          df['reversal_regime_probability'] * 0.3 + 
                          df['high_volume_signal_enhancement'] * 0.3)
    
    # Low Volume Dispersion Signals  
    low_volume_signals = (df['decay_asymmetry_volume_concentration'] * 0.5 + 
                         df['low_volume_signal_suppression'] * 0.5)
    
    # Regime Transition Warning Signals
    transition_signals = (df['decay_regime_precursor'] * 0.6 + 
                         df['breakout_failure'] * 0.2 + 
                         df['breakdown_failure'] * 0.2)
    
    # Dynamic Factor Allocation with regime-dependent weighting
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    alpha_factor[high_volume_mask] = high_volume_signals[high_volume_mask]
    alpha_factor[low_volume_mask] = low_volume_signals[low_volume_mask]
    alpha_factor[transition_mask] = transition_signals[transition_mask]
    
    # Fill remaining periods with weighted combination
    default_mask = ~(high_volume_mask | low_volume_mask | transition_mask)
    alpha_factor[default_mask] = (
        high_volume_signals[default_mask] * 0.3 +
        low_volume_signals[default_mask] * 0.4 +
        transition_signals[default_mask] * 0.3
    )
    
    # Final normalization and cleaning
    alpha_factor = (alpha_factor - alpha_factor.rolling(50).mean()) / (alpha_factor.rolling(50).std() + 1e-6)
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
