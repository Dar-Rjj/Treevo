import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Momentum Asymmetry Persistence Factor
    """
    data = df.copy()
    
    # Calculate Directional Momentum Asymmetry
    # Bullish Momentum Strength
    gap_to_high_extension = (data['high'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8)
    gap_to_high_extension = gap_to_high_extension.replace([np.inf, -np.inf], 0)
    
    daily_range = data['high'] - data['low']
    gap_retention_efficiency = ((data['high'] - data['open']) / (daily_range + 1e-8)).clip(0, 1)
    bullish_strength = gap_to_high_extension * gap_retention_efficiency
    
    # Bearish Momentum Strength
    gap_to_low_extension = (data['open'] - data['low']) / (data['close'].shift(1) - data['open'] + 1e-8)
    gap_to_low_extension = gap_to_low_extension.replace([np.inf, -np.inf], 0)
    
    # Downside momentum persistence (using 3-day rolling pattern)
    downside_pressure = (data['open'] - data['low']).rolling(window=3).mean() / (daily_range.rolling(window=3).mean() + 1e-8)
    bearish_strength = gap_to_low_extension * downside_pressure
    
    # Generate Asymmetry Ratio with directional consistency
    asymmetry_ratio = (bullish_strength + 1e-8) / (bearish_strength + 1e-8)
    directional_consistency = (bullish_strength.rolling(window=5).std() + 1e-8) / (bearish_strength.rolling(window=5).std() + 1e-8)
    weighted_asymmetry = asymmetry_ratio * directional_consistency
    
    # Apply Volume-Weighted Confirmation
    volume_5d_median = data['volume'].rolling(window=5).median()
    volume_surge_ratio = data['volume'] / (volume_5d_median + 1e-8)
    
    # Volume-Price Alignment
    price_momentum = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    volume_price_alignment = np.abs(price_momentum) * volume_surge_ratio
    
    # Volume-Weighted Multiplier
    strong_alignment_mask = volume_price_alignment > volume_price_alignment.rolling(window=10).quantile(0.7)
    weak_alignment_mask = volume_price_alignment < volume_price_alignment.rolling(window=10).quantile(0.3)
    
    volume_multiplier = np.ones_like(data['volume'])
    # Strong alignment: exponential reinforcement
    volume_multiplier[strong_alignment_mask] = np.exp(volume_surge_ratio[strong_alignment_mask] - 1)
    # Weak alignment: linear attenuation with mean reversion
    volume_multiplier[weak_alignment_mask] = 0.5 + 0.5 * (volume_surge_ratio[weak_alignment_mask] - 1)
    
    volume_confirmed_asymmetry = weighted_asymmetry * volume_multiplier
    
    # Compute Momentum Persistence Score
    # Multi-day momentum consistency
    momentum_signals = np.sign(price_momentum.rolling(window=3).mean())
    consecutive_directional = momentum_signals.rolling(window=5).apply(
        lambda x: np.sum(x[:-1] == x[-1]) if len(x) == 5 else 0, raw=True
    )
    
    # Persistence strength with momentum magnitude weighting
    momentum_magnitude = np.abs(price_momentum.rolling(window=5).mean())
    persistence_strength = consecutive_directional * momentum_magnitude
    
    # Persistence adjustment
    strong_persistence_mask = persistence_strength > persistence_strength.rolling(window=20).quantile(0.7)
    weak_persistence_mask = persistence_strength < persistence_strength.rolling(window=20).quantile(0.3)
    
    persistence_adjustment = np.ones_like(persistence_strength)
    # Strong persistence: exponential reinforcement
    persistence_adjustment[strong_persistence_mask] = np.exp(persistence_strength[strong_persistence_mask] / 5)
    # Weak persistence: linear attenuation with contrarian adjustment
    persistence_adjustment[weak_persistence_mask] = 0.8 + 0.2 * (persistence_strength[weak_persistence_mask] / 5)
    
    # Generate Composite Alpha Factor
    composite_factor = volume_confirmed_asymmetry * persistence_adjustment
    
    # Final signal strength classification and scaling
    factor_quantiles = composite_factor.rolling(window=20).apply(
        lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else 1, 
        raw=False
    )
    
    # Apply regime-specific scaling
    strong_signal_mask = factor_quantiles == 2  # Top tercile
    moderate_signal_mask = factor_quantiles == 1  # Middle tercile
    weak_signal_mask = factor_quantiles == 0  # Bottom tercile
    
    final_factor = composite_factor.copy()
    final_factor[strong_signal_mask] = composite_factor[strong_signal_mask] * 1.2
    final_factor[moderate_signal_mask] = composite_factor[moderate_signal_mask] * 1.0
    final_factor[weak_signal_mask] = composite_factor[weak_signal_mask] * 0.8
    
    return final_factor
