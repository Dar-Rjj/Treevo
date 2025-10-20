import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Asymmetric Reversal with Liquidity-Regime Convergence
    """
    data = df.copy()
    
    # 1. Asymmetric Price Reversal Calculation
    # Directional Return Components
    returns = data['close'].pct_change()
    pos_returns_3d = returns.rolling(window=3).apply(lambda x: x[x > 0].sum(), raw=False)
    neg_returns_3d = returns.rolling(window=3).apply(lambda x: x[x < 0].sum(), raw=False)
    asymmetry_ratio = (pos_returns_3d - abs(neg_returns_3d)) / (pos_returns_3d + abs(neg_returns_3d) + 1e-8)
    
    # Multi-Timeframe Reversal
    return_1d = returns
    return_3d = returns.rolling(window=3).sum()
    return_diff = return_1d - return_3d
    return_decay = return_1d / (return_3d + 1e-8)
    magnitude_ratio = abs(return_1d) / (abs(return_3d) + 1e-8)
    
    # Combine Asymmetry and Reversal Strength
    base_reversal = asymmetry_ratio * return_decay * magnitude_ratio
    
    # 2. Volume-Weighted Confirmation System
    # Volume Surge and Persistence Analysis
    volume_accel = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # Volume persistence (consecutive volume increases)
    volume_increase = data['volume'] > data['volume'].shift(1)
    volume_persistence = volume_increase.rolling(window=5).apply(
        lambda x: max(sum(1 for i in range(len(x)) if all(x[:i+1])), 0), raw=False
    )
    volume_surge = data['volume'] / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Amount-Based Order Flow Context
    amount_accel = data['amount'] / (data['amount'].shift(1) + 1e-8)
    amount_volume_ratio = data['amount'] / (data['volume'] + 1e-8)
    amount_volume_ratio_change = amount_volume_ratio.pct_change()
    
    # Large order concentration
    amount_std = data['amount'].rolling(window=5).std()
    amount_mean = data['amount'].rolling(window=5).mean()
    order_concentration = amount_std / (amount_mean + 1e-8)
    
    # Liquidity Momentum Integration
    liquidity_momentum = (volume_accel + amount_accel) / 2
    persistence_filter = np.where(volume_persistence >= 3, 1.2, 1.0)
    liquidity_weight = liquidity_momentum * persistence_filter * order_concentration
    
    # 3. Volatility-Regime Adaptive Filtering
    # Volatility Environment Assessment
    price_range = (data['high'] - data['low']) / data['close']
    range_ratio = price_range / (price_range.rolling(window=20).mean() + 1e-8)
    
    daily_abs_return = abs(returns)
    return_vol_ratio = daily_abs_return / (daily_abs_return.rolling(window=10).mean() + 1e-8)
    
    # Volatility asymmetry
    upside_vol = returns[returns > 0].rolling(window=10).std()
    downside_vol = abs(returns[returns < 0]).rolling(window=10).std()
    vol_asymmetry = upside_vol / (downside_vol + 1e-8)
    
    # Multi-Dimensional Regime Classification
    price_vol_regime = np.where(range_ratio > 1.2, 'High', 
                               np.where(range_ratio < 0.8, 'Low', 'Normal'))
    vol_asym_regime = np.where(vol_asymmetry > 1.5, 'Bull',
                              np.where(vol_asymmetry < 0.67, 'Bear', 'Balanced'))
    
    # Regime-Specific Signal Enhancement
    regime_enhancement = np.ones_like(base_reversal)
    
    # High volatility + Bull asymmetry
    high_bull_mask = (price_vol_regime == 'High') & (vol_asym_regime == 'Bull')
    regime_enhancement[high_bull_mask] = 1.5
    
    # High volatility + Bear asymmetry
    high_bear_mask = (price_vol_regime == 'High') & (vol_asym_regime == 'Bear')
    regime_enhancement[high_bear_mask] = 0.7
    
    # Low volatility periods
    low_vol_mask = price_vol_regime == 'Low'
    regime_enhancement[low_vol_mask] = 1.0 + (volume_persistence[low_vol_mask] * 0.1)
    
    # 4. Multi-Timeframe Convergence Detection
    # Short vs Medium-Term Alignment
    reversal_1d = asymmetry_ratio * return_1d * magnitude_ratio
    reversal_3d = asymmetry_ratio * return_3d * magnitude_ratio
    
    convergence_score = np.where(
        (reversal_1d * reversal_3d) > 0, 
        abs(reversal_1d + reversal_3d) / 2,
        -abs(reversal_1d - reversal_3d) / 2
    )
    
    # Volume-Price Timing Synchronization
    volume_peak_alignment = np.where(
        (volume_accel > 1.2) & (abs(returns) > abs(returns).rolling(window=10).mean()),
        1.2, 1.0
    )
    
    # Liquidity-Regime Coherence Check
    regime_stability = (
        (range_ratio.rolling(window=3).std() < 0.1) &
        (vol_asymmetry.rolling(window=3).std() < 0.2)
    )
    coherence_multiplier = np.where(regime_stability, 1.1, 0.9)
    
    # 5. Dynamic Factor Integration
    # Core Asymmetric Reversal Foundation
    core_factor = base_reversal * return_decay * magnitude_ratio
    
    # Volume-Weighted Enhancement
    volume_weighted = core_factor * volume_surge
    volume_weighted = np.where(volume_persistence >= 2, 
                              volume_weighted * amount_accel,
                              volume_weighted)
    volume_weighted = volume_weighted * order_concentration
    
    # Volatility-Regime Adaptive Filtering
    regime_adjusted = volume_weighted * regime_enhancement
    regime_adjusted = regime_adjusted * vol_asymmetry
    
    # Multi-Timeframe Convergence Optimization
    convergence_boost = np.where(convergence_score > 0, 1.1, 0.9)
    timeframe_aligned = regime_adjusted * convergence_boost * volume_peak_alignment * coherence_multiplier
    
    # Final Alpha Factor
    alpha_factor = timeframe_aligned * liquidity_weight
    
    return pd.Series(alpha_factor, index=data.index)
