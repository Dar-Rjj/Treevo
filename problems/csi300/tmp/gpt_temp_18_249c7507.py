import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor incorporating dynamic multi-dimensional signal integration,
    regime-adaptive scaling, and sophisticated price-volume-amount relationships.
    
    Economic intuition:
    - Dynamic Signal Hierarchy: Multi-timeframe signals with performance-based dynamic weighting
    - Regime-Adaptive Scaling Matrix: Multi-dimensional volatility regime detection and signal adjustment
    - Price-Volume-Amount Efficiency Triad: Measures efficiency across three complementary dimensions
    - Adaptive Signal Integration: Dynamic weighting based on current market conditions and signal effectiveness
    """
    
    # 1. Dynamic Multi-Timeframe Signal Hierarchy
    # Price momentum across multiple horizons with dynamic persistence weighting
    momentum_signals = {}
    momentum_effectiveness = {}
    
    for period in [1, 2, 3, 5, 8]:
        # Raw momentum signals
        momentum_signals[f'mom_{period}d'] = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-7)
        
        # Signal effectiveness based on recent predictive power
        future_returns = df['close'].shift(-1) / df['close'] - 1
        signal_correlation = momentum_signals[f'mom_{period}d'].rolling(window=10).corr(future_returns)
        momentum_effectiveness[period] = np.abs(signal_correlation)
    
    # Dynamic weighting based on signal effectiveness and persistence
    momentum_weights = {}
    total_effectiveness = sum(momentum_effectiveness.values())
    for period in momentum_effectiveness:
        momentum_weights[period] = momentum_effectiveness[period] / (total_effectiveness + 1e-7)
    
    # Combined momentum with dynamic weighting
    combined_momentum = sum(momentum_signals[f'mom_{period}d'] * momentum_weights[period] for period in [1, 2, 3, 5, 8])
    
    # 2. Multi-Dimensional Regime-Adaptive Scaling Matrix
    # Three-dimensional volatility regime detection
    price_range_vol = (df['high'] - df['low']) / (df['close'] + 1e-7)
    close_return_vol = df['close'].pct_change().abs()
    gap_volatility = (df['open'] - df['close'].shift(1)).abs() / (df['close'].shift(1) + 1e-7)
    
    # Multi-timeframe regime classification
    regime_components = {}
    for window in [3, 5, 8, 13]:
        regime_components[f'short_{window}d'] = (
            price_range_vol.rolling(window=window).mean() / 
            (price_range_vol.rolling(window=window*2).mean() + 1e-7)
        )
        regime_components[f'medium_{window}d'] = (
            close_return_vol.rolling(window=window).mean() / 
            (close_return_vol.rolling(window=window*2).mean() + 1e-7)
        )
        regime_components[f'long_{window}d'] = (
            gap_volatility.rolling(window=window).mean() / 
            (gap_volatility.rolling(window=window*2).mean() + 1e-7)
        )
    
    # Dynamic regime scoring with cross-timeframe consistency
    regime_score = (
        regime_components['short_3d'] * 0.15 +
        regime_components['short_5d'] * 0.20 +
        regime_components['medium_5d'] * 0.25 +
        regime_components['medium_8d'] * 0.20 +
        regime_components['long_8d'] * 0.10 +
        regime_components['long_13d'] * 0.10
    )
    
    # Adaptive scaling based on regime intensity
    vol_scaling = 1.0 / (regime_score + 1e-7)
    
    # 3. Price-Volume-Amount Efficiency Triad
    # Volume dynamics across multiple dimensions
    volume_signals = {}
    for window in [1, 2, 3, 5]:
        volume_signals[f'vol_mom_{window}d'] = (
            df['volume'] / (df['volume'].rolling(window=window).mean() + 1e-7) - 1
        )
    
    # Amount dynamics for monetary flow confirmation
    amount_signals = {}
    for window in [1, 2, 3, 5]:
        amount_signals[f'amt_mom_{window}d'] = (
            df['amount'] / (df['amount'].rolling(window=window).mean() + 1e-7) - 1
        )
    
    # Price efficiency measures
    efficiency_measures = {}
    efficiency_measures['directional_eff'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 1e-7)
    efficiency_measures['capture_eff'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-7)
    efficiency_measures['reversal_potential'] = (df['high'] - df['close']) / ((df['high'] - df['low']) + 1e-7)
    
    # Cross-dimensional alignment scores
    alignment_scores = {}
    for window in [1, 2, 3]:
        alignment_scores[f'price_vol_align_{window}d'] = (
            volume_signals[f'vol_mom_{window}d'] * momentum_signals[f'mom_{window}d']
        )
        alignment_scores[f'price_amt_align_{window}d'] = (
            amount_signals[f'amt_mom_{window}d'] * momentum_signals[f'mom_{window}d']
        )
        alignment_scores[f'vol_amt_align_{window}d'] = (
            volume_signals[f'vol_mom_{window}d'] * amount_signals[f'amt_mom_{window}d']
        )
    
    # Combined efficiency triad with dynamic weighting
    efficiency_triad = (
        efficiency_measures['directional_eff'] * 0.25 +
        efficiency_measures['capture_eff'] * 0.20 +
        efficiency_measures['reversal_potential'] * 0.15 +
        alignment_scores['price_vol_align_2d'] * 0.12 +
        alignment_scores['price_amt_align_2d'] * 0.12 +
        alignment_scores['vol_amt_align_2d'] * 0.08 +
        alignment_scores['price_vol_align_3d'] * 0.04 +
        alignment_scores['price_amt_align_3d'] * 0.04
    )
    
    # 4. Adaptive Signal Integration with Performance Feedback
    # Recent performance tracking for dynamic weighting
    momentum_performance = combined_momentum.rolling(window=5).apply(
        lambda x: np.corrcoef(x, np.arange(len(x)))[0,1] if len(x) > 1 else 0
    )
    efficiency_performance = efficiency_triad.rolling(window=5).apply(
        lambda x: np.corrcoef(x, np.arange(len(x)))[0,1] if len(x) > 1 else 0
    )
    
    # Dynamic regime-based signal integration
    high_regime = regime_score > 1.1
    low_regime = regime_score < 0.9
    normal_regime = ~high_regime & ~low_regime
    
    # Performance-adjusted dynamic weights
    base_momentum_weight = np.where(high_regime, 0.30, np.where(low_regime, 0.50, 0.40))
    base_efficiency_weight = np.where(high_regime, 0.45, np.where(low_regime, 0.35, 0.40))
    base_alignment_weight = 1.0 - base_momentum_weight - base_efficiency_weight
    
    # Performance feedback adjustment
    momentum_weight = base_momentum_weight * (1 + momentum_performance * 0.3)
    efficiency_weight = base_efficiency_weight * (1 + efficiency_performance * 0.3)
    alignment_weight = base_alignment_weight * (1 - (momentum_performance + efficiency_performance) * 0.15)
    
    # Weight normalization
    total_weight = momentum_weight + efficiency_weight + alignment_weight
    momentum_weight = momentum_weight / (total_weight + 1e-7)
    efficiency_weight = efficiency_weight / (total_weight + 1e-7)
    alignment_weight = alignment_weight / (total_weight + 1e-7)
    
    # Final alpha factor with adaptive integration
    alpha_factor = (
        combined_momentum * vol_scaling * momentum_weight +
        efficiency_triad * efficiency_weight +
        (alignment_scores['price_vol_align_2d'] + alignment_scores['price_amt_align_2d']) * alignment_weight
    )
    
    return alpha_factor
