import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Price Efficiency Momentum Divergence factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Exponential Moving Average calculations
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    # Price momentum components
    ema_close_3 = ema(data['close'], 3)
    ema_close_6 = ema(data['close'], 6)
    ema_close_10 = ema(data['close'], 10)
    ema_close_20 = ema(data['close'], 20)
    ema_close_40 = ema(data['close'], 40)
    
    price_momentum_3d = ema_close_3 - ema_close_6
    price_momentum_10d = ema_close_10 - ema_close_20
    price_momentum_20d = ema_close_20 - ema_close_40
    
    # Volume momentum components
    ema_volume_3 = ema(data['volume'], 3)
    ema_volume_6 = ema(data['volume'], 6)
    ema_volume_10 = ema(data['volume'], 10)
    ema_volume_20 = ema(data['volume'], 20)
    ema_volume_40 = ema(data['volume'], 40)
    
    volume_momentum_3d = ema_volume_3 - ema_volume_6
    volume_momentum_10d = ema_volume_10 - ema_volume_20
    volume_momentum_20d = ema_volume_20 - ema_volume_40
    
    # Range-Based Efficiency Measures
    # Price efficiency components
    daily_price_efficiency = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    daily_price_efficiency = daily_price_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    price_efficiency_5d = daily_price_efficiency.rolling(window=5, min_periods=3).mean()
    price_efficiency_10d = daily_price_efficiency.rolling(window=10, min_periods=5).mean()
    
    # Volume efficiency components
    daily_volume_efficiency = abs(data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1))
    daily_volume_efficiency = daily_volume_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    volume_efficiency_5d = daily_volume_efficiency.rolling(window=5, min_periods=3).mean()
    volume_efficiency_10d = daily_volume_efficiency.rolling(window=10, min_periods=5).mean()
    
    # Efficiency-Scaled Momentum Signals
    # Price momentum scaling
    eff_price_3d = price_momentum_3d / (price_efficiency_5d + 0.001)
    eff_price_10d = price_momentum_10d / (price_efficiency_10d + 0.001)
    eff_price_20d = price_momentum_20d / (price_efficiency_10d + 0.001)
    
    # Volume momentum scaling
    eff_volume_3d = volume_momentum_3d / (volume_efficiency_5d + 0.001)
    eff_volume_10d = volume_momentum_10d / (volume_efficiency_10d + 0.001)
    eff_volume_20d = volume_momentum_20d / (volume_efficiency_10d + 0.001)
    
    # Multi-Timeframe Divergence Analysis
    # Direct momentum divergence
    divergence_3d = eff_price_3d - eff_volume_3d
    divergence_10d = eff_price_10d - eff_volume_10d
    divergence_20d = eff_price_20d - eff_volume_20d
    
    # Cross-timeframe acceleration
    cross_divergence_sm = (eff_price_3d - eff_price_10d) - (eff_volume_3d - eff_volume_10d)
    cross_divergence_ml = (eff_price_10d - eff_price_20d) - (eff_volume_10d - eff_volume_20d)
    
    # Dynamic Signal Weighting
    # Timeframe emphasis based on efficiency
    price_efficiency_level = price_efficiency_10d.rolling(window=20, min_periods=10).apply(
        lambda x: 0.6 if x.iloc[-1] > x.quantile(0.7) else (0.3 if x.iloc[-1] > x.quantile(0.3) else 0.1)
    )
    
    # Volume confirmation weighting
    volume_momentum_strength = (volume_momentum_10d / volume_momentum_10d.rolling(window=20, min_periods=10).std()).abs()
    volume_weight = np.tanh(volume_momentum_strength * 0.5)
    
    # Weighted average of timeframe divergences
    timeframe_weights = pd.DataFrame({
        'short_term': price_efficiency_level.apply(lambda x: 0.6 if x > 0.5 else 0.2),
        'medium_term': price_efficiency_level.apply(lambda x: 0.3 if 0.3 <= x <= 0.5 else 0.6),
        'long_term': price_efficiency_level.apply(lambda x: 0.1 if x > 0.5 else 0.2)
    })
    
    # Normalize weights to sum to 1
    timeframe_weights = timeframe_weights.div(timeframe_weights.sum(axis=1), axis=0)
    
    weighted_divergence = (
        timeframe_weights['short_term'] * divergence_3d +
        timeframe_weights['medium_term'] * divergence_10d +
        timeframe_weights['long_term'] * divergence_20d
    )
    
    # Cross-timeframe consistency adjustment
    consistency_score = (
        np.sign(divergence_3d) * np.sign(divergence_10d) * 0.5 +
        np.sign(divergence_10d) * np.sign(divergence_20d) * 0.5
    )
    
    # Volume confirmation filter
    volume_direction_alignment = (
        np.sign(volume_momentum_10d) * np.sign(weighted_divergence) * 0.7 +
        np.sign(volume_momentum_20d) * np.sign(weighted_divergence) * 0.3
    )
    
    volume_efficiency_threshold = (volume_efficiency_10d > volume_efficiency_10d.rolling(window=20, min_periods=10).quantile(0.3)).astype(float)
    
    # Final alpha composition
    core_divergence = weighted_divergence * (1 + consistency_score * 0.3)
    volume_filtered = core_divergence * volume_direction_alignment * volume_efficiency_threshold
    volume_weighted = volume_filtered * (1 + volume_weight * 0.5)
    
    # Cross-divergence enhancement
    cross_divergence_enhancement = (cross_divergence_sm + cross_divergence_ml) * 0.25
    
    # Final alpha factor
    alpha_factor = volume_weighted + cross_divergence_enhancement
    
    # Normalize and clean
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=60, min_periods=30).mean()) / alpha_factor.rolling(window=60, min_periods=30).std()
    
    return alpha_factor
