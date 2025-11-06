import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Extended Timeframe Momentum with Multi-Scale Divergence Detection and Volatility Regime Adaptation
    Uses ultra-long timeframes (90-day, 180-day) for momentum calculation to capture persistent trends
    Detects divergences across multiple scales: price-volume, price-amount, and volatility regimes
    Applies sophisticated volatility adjustments using rolling volatility percentiles
    Incorporates regime-dependent momentum weighting and divergence penalties
    """
    # Ultra-long timeframe momentum components for persistent trend capture
    momentum_30d = df['close'] / df['close'].shift(30) - 1
    momentum_60d = df['close'] / df['close'].shift(60) - 1
    momentum_90d = df['close'] / df['close'].shift(90) - 1
    momentum_180d = df['close'] / df['close'].shift(180) - 1
    
    # Multi-scale volume trend analysis
    volume_ma_30 = df['volume'].rolling(window=30, min_periods=1).mean()
    volume_ma_60 = df['volume'].rolling(window=60, min_periods=1).mean()
    volume_ma_90 = df['volume'].rolling(window=90, min_periods=1).mean()
    
    volume_trend_30d = df['volume'] / volume_ma_30 - 1
    volume_trend_60d = df['volume'] / volume_ma_60 - 1
    volume_trend_90d = df['volume'] / volume_ma_90 - 1
    
    # Multi-scale amount trend analysis
    amount_ma_30 = df['amount'].rolling(window=30, min_periods=1).mean()
    amount_ma_60 = df['amount'].rolling(window=60, min_periods=1).mean()
    amount_ma_90 = df['amount'].rolling(window=90, min_periods=1).mean()
    
    amount_trend_30d = df['amount'] / amount_ma_30 - 1
    amount_trend_60d = df['amount'] / amount_ma_60 - 1
    amount_trend_90d = df['amount'] / amount_ma_90 - 1
    
    # Advanced volatility regime detection using true range percentiles
    true_range = pd.DataFrame({
        'high_low': df['high'] - df['low'],
        'high_close_prev': abs(df['high'] - df['close'].shift(1)),
        'low_close_prev': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    normalized_true_range = true_range / df['close']
    
    # Rolling volatility percentiles for regime detection
    volatility_30d = normalized_true_range.rolling(window=30, min_periods=1).mean()
    volatility_60d = normalized_true_range.rolling(window=60, min_periods=1).mean()
    volatility_90d = normalized_true_range.rolling(window=90, min_periods=1).mean()
    
    volatility_percentile_30d = volatility_30d.rolling(window=90, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float), raw=False
    )
    volatility_percentile_60d = volatility_60d.rolling(window=90, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float), raw=False
    )
    
    # Multi-scale price-volume divergence detection
    price_direction_30d = np.sign(momentum_30d)
    price_direction_60d = np.sign(momentum_60d)
    price_direction_90d = np.sign(momentum_90d)
    
    volume_direction_30d = np.sign(volume_trend_30d)
    volume_direction_60d = np.sign(volume_trend_60d)
    volume_direction_90d = np.sign(volume_trend_90d)
    
    pv_divergence_30d = (price_direction_30d != volume_direction_30d).astype(float) * abs(volume_trend_30d)
    pv_divergence_60d = (price_direction_60d != volume_direction_60d).astype(float) * abs(volume_trend_60d)
    pv_divergence_90d = (price_direction_90d != volume_direction_90d).astype(float) * abs(volume_trend_90d)
    
    # Multi-scale price-amount divergence detection
    amount_direction_30d = np.sign(amount_trend_30d)
    amount_direction_60d = np.sign(amount_trend_60d)
    amount_direction_90d = np.sign(amount_trend_90d)
    
    pa_divergence_30d = (price_direction_30d != amount_direction_30d).astype(float) * abs(amount_trend_30d)
    pa_divergence_60d = (price_direction_60d != amount_direction_60d).astype(float) * abs(amount_trend_60d)
    pa_divergence_90d = (price_direction_90d != amount_direction_90d).astype(float) * abs(amount_trend_90d)
    
    # Volatility regime divergence detection
    volatility_trend_30d = np.sign(volatility_30d - volatility_30d.rolling(window=15, min_periods=1).mean())
    volatility_trend_60d = np.sign(volatility_60d - volatility_60d.rolling(window=30, min_periods=1).mean())
    
    pvol_divergence_30d = (price_direction_30d != volatility_trend_30d).astype(float) * abs(volatility_trend_30d)
    pvol_divergence_60d = (price_direction_60d != volatility_trend_60d).astype(float) * abs(volatility_trend_60d)
    
    # Comprehensive divergence aggregation
    total_divergence = (
        pv_divergence_30d + pv_divergence_60d + pv_divergence_90d +
        pa_divergence_30d + pa_divergence_60d + pa_divergence_90d +
        pvol_divergence_30d + pvol_divergence_60d
    ) / 8.0
    
    # Regime-dependent momentum weighting
    high_vol_regime = (volatility_percentile_30d + volatility_percentile_60d) / 2
    
    # High volatility regime favors shorter timeframes, low volatility favors longer timeframes
    momentum_weights_high_vol = [0.4, 0.3, 0.2, 0.1]  # 30d, 60d, 90d, 180d
    momentum_weights_low_vol = [0.2, 0.25, 0.3, 0.25]  # 30d, 60d, 90d, 180d
    
    combined_momentum = (
        high_vol_regime * (
            momentum_weights_high_vol[0] * momentum_30d +
            momentum_weights_high_vol[1] * momentum_60d +
            momentum_weights_high_vol[2] * momentum_90d +
            momentum_weights_high_vol[3] * momentum_180d
        ) +
        (1 - high_vol_regime) * (
            momentum_weights_low_vol[0] * momentum_30d +
            momentum_weights_low_vol[1] * momentum_60d +
            momentum_weights_low_vol[2] * momentum_90d +
            momentum_weights_low_vol[3] * momentum_180d
        )
    )
    
    # Multi-timeframe volatility adjustment
    volatility_adjusted_momentum_30d = combined_momentum / (volatility_30d + 1e-7)
    volatility_adjusted_momentum_60d = combined_momentum / (volatility_60d + 1e-7)
    volatility_adjusted_momentum_90d = combined_momentum / (volatility_90d + 1e-7)
    
    # Regime-dependent volatility adjustment weighting
    volatility_adjusted_momentum = (
        high_vol_regime * (
            0.5 * volatility_adjusted_momentum_30d +
            0.3 * volatility_adjusted_momentum_60d +
            0.2 * volatility_adjusted_momentum_90d
        ) +
        (1 - high_vol_regime) * (
            0.3 * volatility_adjusted_momentum_30d +
            0.4 * volatility_adjusted_momentum_60d +
            0.3 * volatility_adjusted_momentum_90d
        )
    )
    
    # Amount confirmation strength with multi-scale weighting
    amount_confirmation = (
        0.4 * amount_trend_30d +
        0.35 * amount_trend_60d +
        0.25 * amount_trend_90d
    )
    
    # Final factor with regime-adaptive divergence penalties and confirmation signals
    divergence_penalty_strength = 0.4 * high_vol_regime + 0.2 * (1 - high_vol_regime)
    confirmation_strength = 0.15 * high_vol_regime + 0.25 * (1 - high_vol_regime)
    
    factor = (
        volatility_adjusted_momentum * (1 - divergence_penalty_strength * total_divergence) +
        confirmation_strength * amount_confirmation
    )
    
    return factor
