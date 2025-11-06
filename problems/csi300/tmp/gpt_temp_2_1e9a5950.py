import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Price-Volume Divergence factor
    Combines regime classification with price-volume dynamics for adaptive signal generation
    """
    data = df.copy()
    
    # 1. Regime Classification
    # Volatility clustering with GARCH-like persistence
    returns = data['close'].pct_change()
    volatility_short = returns.rolling(window=5).std()
    volatility_medium = returns.rolling(window=20).std()
    volatility_ratio = volatility_short / volatility_medium
    
    # Regime classification based on volatility clustering
    high_vol_regime = (volatility_ratio > 1.2) & (volatility_short > volatility_short.rolling(50).quantile(0.7))
    low_vol_regime = (volatility_ratio < 0.8) & (volatility_short < volatility_short.rolling(50).quantile(0.3))
    
    # Volume-price co-movement patterns
    volume_ma = data['volume'].rolling(window=20).mean()
    volume_spike = data['volume'] > volume_ma * 1.5
    price_range = (data['high'] - data['low']) / data['close']
    low_vol_volume_spike = low_vol_regime & volume_spike & (price_range < price_range.rolling(20).quantile(0.3))
    
    # 2. Price-Volume Dynamics
    # Intraday pressure imbalance
    high_close_pressure = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    close_low_pressure = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    pressure_imbalance = high_close_pressure - close_low_pressure
    
    # Volume-weighted pressure differential
    volume_weighted_pressure = pressure_imbalance * data['volume']
    vwp_ma = volume_weighted_pressure.rolling(window=10).mean()
    vwp_std = volume_weighted_pressure.rolling(window=10).std()
    normalized_vwp = (volume_weighted_pressure - vwp_ma) / (vwp_std + 1e-8)
    
    # Accumulation/distribution patterns
    range_efficiency = (data['close'] - data['open']).abs() / (data['high'] - data['low'] + 1e-8)
    volume_extremes_ratio = data['volume'].rolling(window=5).apply(
        lambda x: (x > x.quantile(0.8)).sum() / (x > x.quantile(0.2)).sum() if len(x) == 5 else np.nan
    )
    
    # 3. Adaptive Signal Generation
    # Regime-dependent weighting
    price_momentum = data['close'].pct_change(periods=5)
    volume_confirmation = data['volume'].pct_change(periods=3)
    
    # High volatility: price momentum emphasis
    high_vol_signal = price_momentum * 0.7 + volume_confirmation * 0.3
    
    # Low volatility: volume confirmation emphasis
    low_vol_signal = price_momentum * 0.3 + volume_confirmation * 0.7
    
    # Multi-scale divergence detection
    short_term_signal = normalized_vwp.rolling(window=5).mean()
    medium_term_signal = normalized_vwp.rolling(window=15).mean()
    signal_discrepancy = short_term_signal - medium_term_signal
    
    # Convergence/divergence acceleration
    discrepancy_change = signal_discrepancy.diff(3)
    
    # 4. Final factor combination
    # Base signal from pressure imbalance
    base_factor = normalized_vwp * range_efficiency
    
    # Regime-adaptive enhancement
    regime_enhancement = np.where(
        high_vol_regime, 
        high_vol_signal * 1.2,
        np.where(
            low_vol_regime,
            low_vol_signal * 1.2 + low_vol_volume_spike.astype(float) * 0.5,
            1.0
        )
    )
    
    # Multi-scale confirmation
    multi_scale_confirmation = signal_discrepancy * discrepancy_change
    
    # Final factor
    factor = base_factor * regime_enhancement * (1 + multi_scale_confirmation)
    
    # Normalize and clean
    factor_ma = factor.rolling(window=20).mean()
    factor_std = factor.rolling(window=20).std()
    normalized_factor = (factor - factor_ma) / (factor_std + 1e-8)
    
    # Remove extreme outliers
    factor_q1 = normalized_factor.rolling(window=50).quantile(0.05)
    factor_q3 = normalized_factor.rolling(window=50).quantile(0.95)
    clipped_factor = np.clip(normalized_factor, factor_q1, factor_q3)
    
    return clipped_factor
