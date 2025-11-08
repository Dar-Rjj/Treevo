import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Multi-Dimensional Confirmation alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Core Momentum Framework
    # Multi-timeframe momentum calculation
    momentum_ultra_short = df['close'] / df['close'].shift(2) - 1
    momentum_short = df['close'] / df['close'].shift(5) - 1
    momentum_medium = df['close'] / df['close'].shift(10) - 1
    momentum_long = df['close'] / df['close'].shift(20) - 1
    
    # Momentum quality assessment
    momentum_signals = pd.DataFrame({
        'ultra_short': momentum_ultra_short > 0,
        'short': momentum_short > 0,
        'medium': momentum_medium > 0,
        'long': momentum_long > 0
    })
    momentum_consistency = momentum_signals.sum(axis=1)
    momentum_magnitude = pd.concat([momentum_ultra_short.abs(), momentum_short.abs(), 
                                  momentum_medium.abs(), momentum_long.abs()], axis=1).mean(axis=1)
    
    # Volatility Regime Detection
    # Volatility measurement
    recent_returns = df['close'].pct_change().rolling(window=5).apply(lambda x: x.iloc[1:].std(), raw=False)
    baseline_returns = df['close'].pct_change().rolling(window=20).apply(lambda x: x.iloc[1:].std(), raw=False)
    
    # Regime classification
    high_vol_regime = recent_returns > (baseline_returns * 1.5)
    low_vol_regime = recent_returns < (baseline_returns * 0.7)
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Price Trend Strength Analysis
    short_trend = df['close'] / df['close'].shift(5) - 1
    medium_trend = df['close'] / df['close'].shift(10) - 1
    
    # Trend regime classification
    strong_trend = (short_trend.abs() > 0.03) & (medium_trend.abs() > 0.05)
    moderate_trend = (short_trend.abs() > 0.015) | (medium_trend.abs() > 0.025)
    weak_trend = ~(strong_trend | moderate_trend)
    
    # Volume-Price Dynamics
    # Volume momentum analysis
    volume_acceleration = (df['volume'] / df['volume'].shift(3) - 1) - (df['volume'] / df['volume'].shift(10) - 1)
    volume_trend_strength = df['volume'] / df['volume'].shift(10) - 1
    
    # Price-volume alignment
    price_trend_sign = np.sign(short_trend)
    volume_trend_sign = np.sign(volume_trend_strength)
    direction_alignment = price_trend_sign == volume_trend_sign
    magnitude_alignment = short_trend.abs() / (volume_trend_strength.abs() + 1e-6)
    
    # Volume quality assessment
    volume_mean_10 = df['volume'].rolling(window=10).mean()
    volume_std_10 = df['volume'].rolling(window=10).std()
    volume_stability = volume_mean_10 / (volume_std_10 + 1e-6)
    recent_volume_intensity = df['volume'] / volume_mean_10
    
    # Intraday Strength Signals
    # Price efficiency measures
    daily_range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
    close_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    
    # Volume concentration
    intraday_volume_efficiency = df['amount'] / (df['volume'] * (df['high'] - df['low']) + 1e-6)
    high_low_volume_ratio = df['volume'] / (df['high'] - df['low'] + 1e-6)
    
    # Adaptive Factor Construction
    # Regime-based momentum selection
    base_factor = pd.Series(index=df.index, dtype=float)
    base_factor[high_vol_regime] = (momentum_ultra_short * 0.8 + momentum_short * 0.2)[high_vol_regime]
    base_factor[low_vol_regime] = (momentum_medium * 0.6 + momentum_long * 0.4)[low_vol_regime]
    base_factor[normal_vol_regime] = (momentum_short * 0.7 + momentum_medium * 0.3)[normal_vol_regime]
    
    # Volume confirmation adjustment
    volume_adjusted_factor = base_factor.copy()
    
    # Strong volume acceleration multiplier
    strong_volume_accel = volume_acceleration.abs() > volume_acceleration.rolling(window=20).quantile(0.7)
    volume_adjusted_factor[strong_volume_accel] *= (1 + np.minimum(volume_acceleration.abs()[strong_volume_accel], 0.5))
    
    # Volume-price alignment multipliers
    volume_adjusted_factor[direction_alignment & (price_trend_sign > 0)] *= 1.15
    volume_adjusted_factor[direction_alignment & (price_trend_sign < 0)] *= 0.85
    
    # High volume quality multiplier
    high_volume_quality = volume_stability > volume_stability.rolling(window=20).quantile(0.7)
    volume_adjusted_factor[high_volume_quality] *= (1 + np.minimum(volume_stability[high_volume_quality] / 10, 0.2))
    
    # Intraday strength enhancement
    final_alpha = volume_adjusted_factor.copy()
    
    # High price efficiency multiplier
    high_price_efficiency = daily_range_efficiency.abs() > daily_range_efficiency.abs().rolling(window=20).quantile(0.7)
    final_alpha[high_price_efficiency] *= (1 + daily_range_efficiency.abs()[high_price_efficiency] * 0.3)
    
    # Strong close position multiplier
    strong_close_position = (close_strength - 0.5).abs() > (close_strength - 0.5).abs().rolling(window=20).quantile(0.7)
    final_alpha[strong_close_position] *= (1 + (close_strength - 0.5).abs()[strong_close_position] * 0.4)
    
    # Concentrated volume multiplier
    concentrated_volume = intraday_volume_efficiency > intraday_volume_efficiency.rolling(window=20).quantile(0.7)
    final_alpha[concentrated_volume] *= (1 + np.minimum(intraday_volume_efficiency[concentrated_volume] * 10, 0.25))
    
    return final_alpha
