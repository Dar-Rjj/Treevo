import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Scale Regime-Adaptive Gap Momentum Framework
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper functions
    def calculate_atr(window):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def calculate_vwap_intraday():
        """Calculate intraday VWAP approximation"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # Multi-Frequency Regime Identification
    atr_3 = calculate_atr(3)
    atr_10 = calculate_atr(10)
    volatility_regime = atr_3 / atr_10
    
    # Flow regime detection (approximating buy volume)
    flow_regime = (data['amount'] / data['volume']).rolling(window=5).mean() / data['close']
    flow_regime = flow_regime / flow_regime.rolling(window=10).std()
    
    # Gap regime classification
    gap_magnitude = abs(data['close'] - data['open'])
    range_magnitude = data['high'] - data['low']
    gap_regime_corr = gap_magnitude.rolling(window=7).corr(range_magnitude)
    
    # Temporal Gap Pattern Analysis
    # Morning session gap efficiency approximation
    morning_efficiency = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    morning_efficiency = morning_efficiency.rolling(window=3).mean()
    
    # Afternoon gap momentum approximation
    midday_price = (data['high'] + data['low']) / 2
    afternoon_momentum = (data['close'] - midday_price) / (data['high'] - data['low'] + 1e-8)
    afternoon_momentum = afternoon_momentum.rolling(window=3).mean()
    
    # Overnight gap persistence
    overnight_gap = data['open'] - data['close'].shift(1)
    overnight_persistence = overnight_gap.rolling(window=3).apply(lambda x: np.sign(x).sum() / len(x), raw=False)
    
    # Volume-Anchored Gap Dynamics
    vwap_intraday = calculate_vwap_intraday()
    vwap_gap_proximity = (data['close'] - vwap_intraday) / (atr_5 := calculate_atr(5))
    
    # Flow-enhanced gap momentum
    flow_enhanced_momentum = (data['close'] - data['open']) * flow_regime
    
    # Volume-pressure gap alignment
    volume_pressure = (data['volume'] / data['volume'].shift(1)) * abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Anchored gap efficiency
    anchored_gap_efficiency = vwap_gap_proximity * flow_enhanced_momentum
    
    # Cross-Scale Momentum Integration
    micro_gap_momentum = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    macro_gap_momentum = (data['close'] - data['close'].shift(5)) / atr_10
    
    momentum_regime_alignment = micro_gap_momentum * macro_gap_momentum
    regime_adapted_momentum = momentum_regime_alignment * volatility_regime
    
    # Gap Absorption & Rejection Analysis
    gap_absorption_ratio = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    bidirectional_rejection = ((data['high'] - data['close']) - (data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    flow_driven_absorption = gap_absorption_ratio * flow_regime
    rejection_refined_gap = (1 - abs(bidirectional_rejection)) * flow_driven_absorption
    
    # Structural Break Detection
    gap_magnitude_var = gap_magnitude.rolling(window=10).var()
    volatility_break = gap_magnitude_var / gap_magnitude_var.rolling(window=20).mean()
    
    flow_autocorr = flow_regime.rolling(window=10).apply(lambda x: pd.Series(x).autocorr(), raw=False)
    flow_regime_break = abs(flow_autocorr.diff())
    
    overnight_intraday_divergence = abs(overnight_gap.rolling(window=5).std() - morning_efficiency.rolling(window=5).std())
    
    multi_timeframe_break = (volatility_break + flow_regime_break + overnight_intraday_divergence) / 3
    
    # Liquidity-Enhanced Gap Momentum
    intraday_liquidity = data['volume'] / (data['high'] - data['low'] + 1e-8)
    gap_liquidity_efficiency = abs(data['close'] - data['open']) / (intraday_liquidity + 1e-8)
    flow_liquidity_convergence = flow_regime * gap_liquidity_efficiency
    liquidity_adapted_momentum = regime_adapted_momentum * (1 + flow_liquidity_convergence)
    
    # Core Alpha Construction
    regime_weighted_gap = anchored_gap_efficiency * volatility_regime
    momentum_enhanced_factor = regime_weighted_gap * regime_adapted_momentum
    volume_integrated_alpha = momentum_enhanced_factor * volume_pressure
    break_adjusted_signal = volume_integrated_alpha * (1 - multi_timeframe_break.clip(upper=1))
    
    # Adaptive Regime Integration
    high_vol_alpha = break_adjusted_signal * rejection_refined_gap
    normal_alpha = break_adjusted_signal * liquidity_adapted_momentum
    transition_alpha = break_adjusted_signal * flow_liquidity_convergence
    
    # Cross-regime adaptive alpha based on current regime probabilities
    vol_regime_weight = volatility_regime.rolling(window=5).apply(lambda x: (x > 1.2).sum() / len(x), raw=False)
    flow_regime_weight = flow_regime.rolling(window=5).apply(lambda x: (x > x.median()).sum() / len(x), raw=False)
    
    cross_regime_alpha = (
        vol_regime_weight * high_vol_alpha + 
        (1 - vol_regime_weight) * normal_alpha * (1 - flow_regime_weight) +
        flow_regime_weight * transition_alpha
    )
    
    # Final alpha factor with normalization
    alpha_factor = cross_regime_alpha
    alpha_factor = alpha_factor / alpha_factor.rolling(window=20).std()
    
    return alpha_factor
