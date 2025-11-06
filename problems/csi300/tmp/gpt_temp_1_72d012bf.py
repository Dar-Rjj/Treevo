import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Entropic Momentum with Liquidity-Adaptive Microstructure alpha factor
    """
    data = df.copy()
    
    # Entropic Range Decomposition
    # Micro-Entropy (2-day)
    close_diff_micro = data['close'].diff().abs()
    micro_entropy_window = 2
    micro_entropy = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if i >= micro_entropy_window:
            window_data = close_diff_micro.iloc[i-micro_entropy_window+1:i+1]
            total_sum = window_data.sum()
            if total_sum > 0:
                probabilities = window_data / total_sum
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                micro_entropy.iloc[i] = entropy
            else:
                micro_entropy.iloc[i] = 0
        else:
            micro_entropy.iloc[i] = 0
    
    # Meso-Entropy (8-day)
    meso_entropy_window = 8
    meso_entropy = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if i >= meso_entropy_window:
            window_data = close_diff_micro.iloc[i-meso_entropy_window+1:i+1]
            total_sum = window_data.sum()
            if total_sum > 0:
                probabilities = window_data / total_sum
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                meso_entropy.iloc[i] = entropy
            else:
                meso_entropy.iloc[i] = 0
        else:
            meso_entropy.iloc[i] = 0
    
    # Entropic Divergence
    entropic_divergence = micro_entropy - meso_entropy
    
    # Momentum Entropy Integration
    micro_momentum = np.sign(data['close'].diff()) * (1 - micro_entropy)
    meso_momentum = np.sign(data['close'] - data['close'].shift(7)) * (1 - meso_entropy)
    entropic_momentum_gradient = micro_momentum - meso_momentum
    
    # Entropic Regime Classification
    high_entropy = (micro_entropy > 0.8) & (meso_entropy > 0.7)
    low_entropy = (micro_entropy < 0.3) & (meso_entropy < 0.4)
    transition_entropy = ~high_entropy & ~low_entropy
    
    # Liquidity Microstructure Processing
    # Volume-Weighted Spread
    volume_weighted_spread = (data['high'] - data['low']) / (data['volume'] ** (1/3) + 1e-10)
    
    # Amount Concentration
    amount_rolling_sum = data['amount'].rolling(window=4, min_periods=1).sum()
    amount_concentration = data['amount'] / (amount_rolling_sum + 1e-10)
    
    # Liquidity Efficiency
    liquidity_efficiency = close_diff_micro / (volume_weighted_spread + 1e-10)
    
    # Microstructure Regimes
    liquid_regime = (volume_weighted_spread < 0.02) & (amount_concentration < 2.0)
    illiquid_regime = (volume_weighted_spread > 0.05) | (amount_concentration > 3.0)
    normal_regime = ~liquid_regime & ~illiquid_regime
    
    # Regime-Adaptive Entropy Processing
    regime_adaptive_entropy = pd.Series(index=data.index, dtype=float)
    regime_adaptive_entropy[liquid_regime] = entropic_momentum_gradient[liquid_regime] * (1 + liquidity_efficiency[liquid_regime])
    regime_adaptive_entropy[illiquid_regime] = entropic_momentum_gradient[illiquid_regime] / (1 + volume_weighted_spread[illiquid_regime])
    regime_adaptive_entropy[normal_regime] = entropic_momentum_gradient[normal_regime] * amount_concentration[normal_regime]
    
    # Price-Volume Fractal Integration
    # Volume Fractal Dimension
    short_volume_fractal = np.log(data['volume'] + 1e-10) / np.log(data['high'] - data['low'] + 1e-10)
    
    # Medium Volume Fractal
    volume_rolling = data['volume'].rolling(window=5, min_periods=1).sum()
    high_rolling = data['high'].rolling(window=5, min_periods=1).max()
    low_rolling = data['low'].rolling(window=5, min_periods=1).min()
    medium_volume_fractal = np.log(volume_rolling + 1e-10) / np.log(high_rolling - low_rolling + 1e-10)
    
    # Volume Fractal Change
    volume_fractal_change = short_volume_fractal - medium_volume_fractal
    
    # Price-Volume Entropy Coupling
    micro_coupling = micro_entropy * volume_fractal_change
    meso_coupling = meso_entropy * volume_fractal_change
    coupling_divergence = micro_coupling - meso_coupling
    
    # Fractal Entropy Momentum
    high_entropy_fractal = coupling_divergence * entropic_divergence
    low_entropy_fractal = volume_fractal_change * entropic_momentum_gradient
    
    adaptive_fractal_signal = pd.Series(index=data.index, dtype=float)
    adaptive_fractal_signal[high_entropy] = high_entropy_fractal[high_entropy]
    adaptive_fractal_signal[low_entropy | transition_entropy] = low_entropy_fractal[low_entropy | transition_entropy]
    
    # Momentum Acceleration & Reversal
    # Acceleration Metrics
    micro_acceleration = data['close'].diff() - data['close'].diff().shift(1)
    meso_acceleration = (data['close'] - data['close'].shift(3)) - (data['close'].shift(3) - data['close'].shift(6))
    acceleration_ratio = micro_acceleration / (meso_acceleration.abs() + 0.001)
    
    # Reversal Detection
    price_reversal = np.sign(micro_acceleration) != np.sign(meso_acceleration)
    volume_reversal = np.sign(data['volume'].diff()) != np.sign(data['volume'].diff().shift(1))
    combined_reversal = price_reversal & volume_reversal
    
    # Acceleration-Entropy Fusion
    normal_acceleration = acceleration_ratio * (1 - micro_entropy)
    reversal_acceleration = -acceleration_ratio * micro_entropy
    
    adaptive_acceleration = pd.Series(index=data.index, dtype=float)
    adaptive_acceleration[combined_reversal] = reversal_acceleration[combined_reversal]
    adaptive_acceleration[~combined_reversal] = normal_acceleration[~combined_reversal]
    
    # Composite Alpha Factor
    core_entropic_signal = regime_adaptive_entropy
    microstructure_enhancement = core_entropic_signal * (1 + adaptive_fractal_signal * 0.15)
    acceleration_modulation = microstructure_enhancement * (1 + adaptive_acceleration * 0.25)
    final_factor = acceleration_modulation * (1 + entropic_divergence * 0.1)
    
    return final_factor.fillna(0)
