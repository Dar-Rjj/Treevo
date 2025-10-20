import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining multiple market microstructure insights
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # 1. Intraday Momentum Persistence
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    momentum_persistence = intraday_momentum.rolling(window=5).apply(lambda x: x.autocorr(), raw=True)
    volume_confirmation = df['volume'] / df['volume'].rolling(window=20).mean()
    momentum_factor = momentum_persistence * volume_confirmation
    
    # 2. Pressure Accumulation Breakout
    buying_pressure = (df['close'] - df['low']) * df['volume']
    selling_pressure = (df['high'] - df['close']) * df['volume']
    net_pressure_ratio = (buying_pressure - selling_pressure) / (buying_pressure + selling_pressure + 1e-8)
    
    # Exponential pressure accumulation
    pressure_accumulation = net_pressure_ratio.ewm(span=10).mean()
    pressure_change = pressure_accumulation.diff(3)
    amount_confirmation = df['amount'] / df['amount'].rolling(window=20).mean()
    pressure_factor = pressure_change * amount_confirmation
    
    # 3. Volatility Regime Transition
    vol_3d = (df['high'] - df['low']).rolling(window=3).mean() / df['close']
    vol_10d = (df['high'] - df['low']).rolling(window=10).mean() / df['close']
    vol_ratio = vol_3d / (vol_10d + 1e-8)
    
    # Regime transition detection
    regime_change = vol_ratio.rolling(window=5).std()
    regime_persistence = vol_ratio.rolling(window=10).apply(lambda x: len(np.unique(np.diff(x) > 0)) == 1, raw=True)
    volume_regime = df['volume'] / df['volume'].rolling(window=20).std()
    volatility_factor = regime_change * regime_persistence * volume_regime
    
    # 4. Opening Gap Momentum
    gap_strength = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    gap_persistence = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    gap_volume = df['volume'] / df['volume'].rolling(window=10).mean()
    gap_factor = gap_strength * gap_persistence * gap_volume
    
    # 5. Efficiency-Momentum Divergence
    price_range = df['high'] - df['low']
    directional_move = np.abs(df['close'] - df['close'].shift(1))
    efficiency = directional_move / (price_range + 1e-8)
    
    momentum_3d = df['close'].pct_change(3)
    efficiency_3d = efficiency.rolling(window=3).mean()
    divergence = momentum_3d - efficiency_3d
    volume_divergence = df['volume'].pct_change(3) - df['volume'].pct_change(1)
    efficiency_factor = divergence * volume_divergence
    
    # 6. Microstructure Pressure
    opening_pressure = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    closing_pressure = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    pressure_shift = closing_pressure - opening_pressure
    
    pressure_accum = pressure_shift.rolling(window=5).sum()
    pressure_exhaustion = pressure_accum.rolling(window=10).std()
    volume_pressure_corr = df['volume'].rolling(window=10).corr(pressure_shift)
    microstructure_factor = pressure_accum * pressure_exhaustion * volume_pressure_corr
    
    # 7. Trend Acceleration with Volume Alignment
    trend_3d = df['close'].pct_change(3)
    trend_5d = df['close'].pct_change(5)
    acceleration = trend_3d - trend_5d
    
    volume_trend = df['volume'].pct_change(5)
    volume_alignment = np.sign(acceleration) == np.sign(volume_trend)
    composite_signal = acceleration * volume_alignment.astype(float)
    trend_factor = composite_signal * np.abs(acceleration)
    
    # 8. Cumulative Imbalance Oscillator
    daily_bias = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    weighted_imbalance = daily_bias * df['volume']
    
    # Cumulative sum with reset at extremes
    cum_imbalance = weighted_imbalance.rolling(window=10).sum()
    oscillator = (cum_imbalance - cum_imbalance.rolling(window=20).min()) / \
                (cum_imbalance.rolling(window=20).max() - cum_imbalance.rolling(window=20).min() + 1e-8)
    
    # Regime adjustment using volatility
    regime_adjustment = 1 / (vol_10d + 0.1)
    imbalance_factor = oscillator * regime_adjustment
    
    # 9. Intraday Recovery Momentum
    recovery_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    recovery_persistence = (recovery_strength > 0.5).rolling(window=3).sum()
    
    recovery_momentum = recovery_strength * recovery_persistence
    volume_momentum = df['volume'].pct_change(3)
    recovery_factor = recovery_momentum * volume_momentum
    
    # 10. Regime-Sensitive Momentum
    raw_momentum_5d = df['close'].pct_change(5)
    raw_momentum_10d = df['close'].pct_change(10)
    
    # Regime scaling - momentum works better in certain volatility regimes
    regime_scaling = 1 / (1 + np.abs(vol_ratio - 1))
    regime_filter = (vol_ratio > 0.8) & (vol_ratio < 1.2)
    
    volume_regime_confirmation = df['volume'] / df['volume'].rolling(window=20).mean()
    regime_momentum = raw_momentum_5d * regime_scaling * regime_filter.astype(float) * volume_regime_confirmation
    
    # Combine all factors with equal weights
    factors = [
        momentum_factor,
        pressure_factor,
        volatility_factor,
        gap_factor,
        efficiency_factor,
        microstructure_factor,
        trend_factor,
        imbalance_factor,
        recovery_factor,
        regime_momentum
    ]
    
    # Normalize and combine
    normalized_factors = []
    for factor in factors:
        if factor.notna().any():
            normalized = (factor - factor.rolling(window=50).mean()) / (factor.rolling(window=50).std() + 1e-8)
            normalized_factors.append(normalized)
    
    if normalized_factors:
        composite_factor = sum(normalized_factors) / len(normalized_factors)
    else:
        composite_factor = pd.Series(0, index=df.index)
    
    return composite_factor
