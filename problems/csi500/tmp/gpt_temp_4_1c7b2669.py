import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Divergence factor
    Combines multi-timeframe momentum with volatility regime classification
    and microstructure validation for enhanced predictive power
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Framework
    # Price momentum divergence (5-day vs 20-day)
    mom_5 = data['close'].pct_change(5)
    mom_20 = data['close'].pct_change(20)
    price_divergence = mom_5 - mom_20
    
    # Range efficiency momentum (current vs 5-day average)
    daily_range = (data['high'] - data['low']) / data['close']
    avg_range_5 = daily_range.rolling(5).mean()
    range_efficiency = daily_range / avg_range_5
    
    # Volume-weighted range expansion signals
    volume_weighted_range = (data['high'] - data['low']) * data['volume']
    range_expansion = volume_weighted_range / volume_weighted_range.rolling(5).mean()
    
    # Volatility Regime Processing
    # Triple regime classification using range behavior
    range_std = daily_range.rolling(20).std()
    range_mean = daily_range.rolling(20).mean()
    
    # Define regimes: low (0), normal (1), high (2) volatility
    volatility_regime = np.zeros(len(data))
    volatility_regime[daily_range > (range_mean + range_std)] = 2  # High volatility
    volatility_regime[daily_range < (range_mean - range_std)] = 0  # Low volatility
    volatility_regime[(daily_range >= (range_mean - range_std)) & 
                     (daily_range <= (range_mean + range_std))] = 1  # Normal volatility
    
    # Regime-specific signal transformation
    regime_multipliers = {0: 0.7, 1: 1.0, 2: 1.3}  # Conservative in low vol, aggressive in high vol
    
    # Dynamic component weighting by correlation
    corr_window = 10
    price_div_weight = pd.Series(1.0, index=data.index)
    range_eff_weight = pd.Series(1.0, index=data.index)
    range_exp_weight = pd.Series(1.0, index=data.index)
    
    for i in range(corr_window, len(data)):
        window_data = data.iloc[i-corr_window:i]
        
        # Calculate forward returns for correlation
        forward_ret = window_data['close'].pct_change().shift(-1).dropna()
        if len(forward_ret) < 3:
            continue
            
        valid_idx = forward_ret.index
        
        # Correlate each component with forward returns
        price_div_corr = price_divergence.loc[valid_idx].corr(forward_ret)
        range_eff_corr = range_efficiency.loc[valid_idx].corr(forward_ret)
        range_exp_corr = range_expansion.loc[valid_idx].corr(forward_ret)
        
        # Update weights based on recent predictive power
        if not np.isnan(price_div_corr):
            price_div_weight.iloc[i] = abs(price_div_corr)
        if not np.isnan(range_eff_corr):
            range_eff_weight.iloc[i] = abs(range_eff_corr)
        if not np.isnan(range_exp_corr):
            range_exp_weight.iloc[i] = abs(range_exp_corr)
    
    # Normalize weights
    total_weight = price_div_weight + range_eff_weight + range_exp_weight
    price_div_weight_norm = price_div_weight / total_weight
    range_eff_weight_norm = range_eff_weight / total_weight
    range_exp_weight_norm = range_exp_weight / total_weight
    
    # Microstructure Validation
    # Volume clustering pattern confirmation
    volume_ma_5 = data['volume'].rolling(5).mean()
    volume_cluster = data['volume'] / volume_ma_5
    volume_cluster_signal = np.where(volume_cluster > 1.2, 1.0, 
                                   np.where(volume_cluster < 0.8, -0.5, 0.0))
    
    # Range breakout validation
    range_breakout = np.where(data['close'] > data['high'].shift(1), 1.0,
                            np.where(data['close'] < data['low'].shift(1), -1.0, 0.0))
    
    # Gap filling momentum adjustment
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_filling = np.where(
        (overnight_gap > 0.01) & (data['close'] < data['open']), -1.0,
        np.where((overnight_gap < -0.01) & (data['close'] > data['open']), 1.0, 0.0)
    )
    
    # Combine all components with regime adaptation
    base_factor = (
        price_divergence * price_div_weight_norm +
        range_efficiency * range_eff_weight_norm +
        range_expansion * range_exp_weight_norm
    )
    
    # Apply regime-specific multipliers
    regime_adjusted_factor = base_factor.copy()
    for regime, multiplier in regime_multipliers.items():
        regime_mask = volatility_regime == regime
        regime_adjusted_factor[regime_mask] *= multiplier
    
    # Apply microstructure validation adjustments
    final_factor = (
        regime_adjusted_factor * 
        (1 + 0.1 * volume_cluster_signal) *
        (1 + 0.05 * range_breakout) *
        (1 + 0.08 * gap_filling)
    )
    
    # Normalize the final factor
    final_factor = (final_factor - final_factor.rolling(20).mean()) / final_factor.rolling(20).std()
    
    return final_factor
