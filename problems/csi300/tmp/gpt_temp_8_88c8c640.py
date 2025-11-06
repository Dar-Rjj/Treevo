import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate regime-adaptive alpha factor using second-order derivatives and multi-timeframe convergence
    """
    # Price and volume calculations
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # 1. Price Acceleration Momentum
    price_momentum = close.pct_change()
    price_acceleration = price_momentum.diff()
    volume_momentum = volume.pct_change()
    volume_acceleration = volume_momentum.diff()
    price_acc_momentum = price_acceleration * volume_acceleration
    
    # 2. Volatility Acceleration Divergence
    daily_range = (high - low) / close
    range_momentum = daily_range.diff()
    range_acceleration = range_momentum.diff()
    open_close_movement = (close - open_price) / close
    vol_acc_divergence = range_acceleration * open_close_movement
    
    # 3. Volume-Price Curvature Alignment
    price_curvature = price_acceleration.rolling(window=3).apply(
        lambda x: (x.iloc[-1] - 2*x.iloc[1] + x.iloc[0]) if len(x) == 3 else np.nan, raw=False
    )
    volume_curvature = volume_acceleration.rolling(window=3).apply(
        lambda x: (x.iloc[-1] - 2*x.iloc[1] + x.iloc[0]) if len(x) == 3 else np.nan, raw=False
    )
    curvature_alignment = price_curvature * volume_curvature
    
    # 4. Triple-Derivative Momentum Alignment
    acc_5d = price_momentum.rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 5 else np.nan, raw=False
    )
    acc_10d = price_momentum.rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 10 else np.nan, raw=False
    )
    acc_20d = price_momentum.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 20 else np.nan, raw=False
    )
    
    # Volume trend consistency weights
    vol_trend_5d = volume.rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else 0, raw=False
    )
    vol_trend_10d = volume.rolling(window=10).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 10 and not np.isnan(x).any() else 0, raw=False
    )
    vol_trend_20d = volume.rolling(window=20).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 20 and not np.isnan(x).any() else 0, raw=False
    )
    
    momentum_convergence = (
        acc_5d * vol_trend_5d + 
        acc_10d * vol_trend_10d + 
        acc_20d * vol_trend_20d
    ) / 3
    
    # 5. Volatility Regime Adaptive Features
    range_acc_ma = range_acceleration.rolling(window=10).mean()
    range_acc_std = range_acceleration.rolling(window=10).std()
    volatility_regime = (range_acceleration - range_acc_ma) / range_acc_std
    
    # Regime-specific momentum
    high_vol_regime = (volatility_regime > 1).astype(int)
    low_vol_regime = (volatility_regime < -1).astype(int)
    
    regime_momentum = (
        price_acceleration * high_vol_regime * 0.7 +  # Reduced weight in high vol
        price_acceleration * low_vol_regime * 1.3    # Enhanced weight in low vol
    )
    
    # 6. Multi-Scale Volume Confirmation
    vol_short = volume.rolling(window=5).mean().pct_change()
    vol_medium = volume.rolling(window=10).mean().pct_change()
    vol_long = volume.rolling(window=20).mean().pct_change()
    
    volume_convergence = (vol_short + vol_medium + vol_long) / 3
    volume_price_confirmation = price_acceleration * volume_convergence
    
    # 7. Dynamic Mean Reversion Enhancement
    recent_range = (high.rolling(window=5).max() - low.rolling(window=5).min()) / close
    price_deviation = (close - close.rolling(window=5).mean()) / close.rolling(window=5).std()
    
    # Volatility cluster adjustment
    vol_cluster = daily_range.rolling(window=10).std()
    adjusted_reversion = price_deviation / (1 + vol_cluster)
    
    # 8. Momentum Regime Transition Detection
    momentum_phase = price_acceleration.rolling(window=5).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] and x.mean() > 0 else 
                  -1 if x.iloc[-1] < x.iloc[0] and x.mean() < 0 else 0, raw=False
    )
    
    # Volume confirmation across timeframes
    vol_conf_short = volume.pct_change(3).rolling(window=3).mean()
    vol_conf_medium = volume.pct_change(5).rolling(window=5).mean()
    vol_conf_long = volume.pct_change(10).rolling(window=10).mean()
    
    phase_transition = momentum_phase * (vol_conf_short + vol_conf_medium + vol_conf_long) / 3
    
    # 9. Cross-Feature Interaction Signals
    volume_efficiency = (close - open_price) / (high - low) * volume
    volume_efficiency_norm = (volume_efficiency - volume_efficiency.rolling(window=20).mean()) / volume_efficiency.rolling(window=20).std()
    
    cross_interaction = price_curvature * volume_efficiency_norm
    
    # Final signal combination with regime-adaptive weighting
    volatility_weight = 1 / (1 + np.abs(volatility_regime))
    
    final_signal = (
        price_acc_momentum * 0.15 +
        vol_acc_divergence * 0.12 +
        curvature_alignment * 0.18 +
        momentum_convergence * 0.20 +
        regime_momentum * 0.10 +
        volume_price_confirmation * 0.08 +
        adjusted_reversion * 0.07 +
        phase_transition * 0.06 +
        cross_interaction * 0.04
    ) * volatility_weight
    
    return pd.Series(final_signal, index=df.index, name='alpha_factor')
