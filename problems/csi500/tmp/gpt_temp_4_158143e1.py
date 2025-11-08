import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Divergence with Volume Acceleration alpha factor
    """
    # Multi-Timeframe Momentum Calculation
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Price momentum calculations
    price_momentum_5 = close / close.shift(5) - 1
    price_momentum_10 = close / close.shift(10) - 1
    price_momentum_20 = close / close.shift(20) - 1
    
    # Volume momentum calculations
    volume_momentum_5 = volume / volume.shift(5) - 1
    volume_momentum_10 = volume / volume.shift(10) - 1
    volume_momentum_20 = volume / volume.shift(20) - 1
    
    # Exponential Smoothing Application
    alpha = 0.3
    
    # Apply EMA to momentum series
    price_ema_5 = price_momentum_5.ewm(alpha=alpha).mean()
    price_ema_10 = price_momentum_10.ewm(alpha=alpha).mean()
    price_ema_20 = price_momentum_20.ewm(alpha=alpha).mean()
    
    volume_ema_5 = volume_momentum_5.ewm(alpha=alpha).mean()
    volume_ema_10 = volume_momentum_10.ewm(alpha=alpha).mean()
    volume_ema_20 = volume_momentum_20.ewm(alpha=alpha).mean()
    
    # Calculate acceleration terms
    price_accel_5 = price_ema_5 - price_ema_5.shift(1)
    price_accel_10 = price_ema_10 - price_ema_10.shift(1)
    price_accel_20 = price_ema_20 - price_ema_20.shift(1)
    
    volume_accel_5 = volume_ema_5 - volume_ema_5.shift(1)
    volume_accel_10 = volume_ema_10 - volume_ema_10.shift(1)
    volume_accel_20 = volume_ema_20 - volume_ema_20.shift(1)
    
    # Regime-Aware Weighting
    # Volatility regime detection
    daily_range = (high - low) / close
    volatility_20d = daily_range.rolling(window=20).mean()
    volatility_percentile = volatility_20d.rolling(window=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Classify volatility regimes
    high_vol_regime = (volatility_percentile > 0.7).astype(int)
    low_vol_regime = (volatility_percentile < 0.3).astype(int)
    medium_vol_regime = ((volatility_percentile >= 0.3) & (volatility_percentile <= 0.7)).astype(int)
    
    # Timeframe weight assignment
    short_weights = high_vol_regime * 0.6 + medium_vol_regime * 0.3 + low_vol_regime * 0.1
    medium_weights = high_vol_regime * 0.3 + medium_vol_regime * 0.4 + low_vol_regime * 0.3
    long_weights = high_vol_regime * 0.1 + medium_vol_regime * 0.3 + low_vol_regime * 0.6
    
    # Divergence Pattern Recognition
    # Signal strength calculation for each timeframe
    def calculate_divergence_strength(price_accel, volume_accel):
        bullish = ((price_accel > 0) & (volume_accel > price_accel)).astype(int) * (volume_accel - price_accel)
        bearish = ((price_accel < 0) & (volume_accel < price_accel)).astype(int) * (price_accel - volume_accel)
        return bullish + bearish
    
    divergence_5 = calculate_divergence_strength(price_accel_5, volume_accel_5)
    divergence_10 = calculate_divergence_strength(price_accel_10, volume_accel_10)
    divergence_20 = calculate_divergence_strength(price_accel_20, volume_accel_20)
    
    # Multi-timeframe consistency assessment
    timeframe_alignment = (
        np.sign(divergence_5) * np.sign(divergence_10) * np.sign(divergence_20)
    )
    alignment_multiplier = np.where(timeframe_alignment > 0, 1.5, 1.0)
    
    # Final Alpha Factor Construction
    # Combine weighted signals
    weighted_signal = (
        divergence_5 * short_weights +
        divergence_10 * medium_weights +
        divergence_20 * long_weights
    ) * alignment_multiplier
    
    # Apply exponential smoothing to final signal
    final_signal = weighted_signal.ewm(alpha=0.15).mean()
    
    # Ensure stationarity through differencing
    stationary_signal = final_signal - final_signal.shift(1)
    
    # Remove extreme outliers
    signal_std = stationary_signal.std()
    stationary_signal = np.clip(stationary_signal, -3 * signal_std, 3 * signal_std)
    
    return stationary_signal
