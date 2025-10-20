import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Convergence with Volatility-Regime and Squeeze Emphasis
    """
    data = df.copy()
    
    # Multi-Period Momentum with Exponential Decay
    # Calculate returns across multiple horizons
    returns_5d = data['close'].pct_change(5)
    returns_20d = data['close'].pct_change(20)
    returns_60d = data['close'].pct_change(60)
    
    # Apply exponential decay weighting (λ = 0.94)
    lambda_val = 0.94
    decay_weights = np.array([lambda_val**5, lambda_val**20, lambda_val**60])
    decay_momentum = (returns_5d * decay_weights[0] + 
                     returns_20d * decay_weights[1] + 
                     returns_60d * decay_weights[2])
    
    # Volume Momentum and Convergence
    # Volume momentum calculation
    volume_5d = data['volume'].rolling(5).mean()
    volume_20d = data['volume'].rolling(20).mean()
    volume_ratio = volume_5d / volume_20d
    volume_acceleration = volume_ratio.diff(5)
    
    # Large Order Emphasis
    amount_5d = data['amount'].rolling(5).mean()
    volume_5d_amt = data['volume'].rolling(5).mean()
    amount_volume_ratio = amount_5d / volume_5d_amt
    
    # Volume-directional alignment with price momentum
    volume_momentum_alignment = np.sign(returns_5d) * volume_acceleration
    
    # Convergence signal
    momentum_acceleration = returns_5d.diff(5)
    convergence_signal = momentum_acceleration * volume_ratio
    
    # Volatility Regime and Squeeze Detection
    # Volatility Regime Classification
    returns_10d_std = data['close'].pct_change().rolling(10).std()
    returns_20d_std = data['close'].pct_change().rolling(20).std()
    volatility_ratio = returns_10d_std / returns_20d_std
    
    # Regime classification
    regime = pd.Series(1.0, index=data.index)  # Normal regime
    regime[volatility_ratio > 1.2] = 0.7       # High volatility
    regime[volatility_ratio < 0.8] = 1.3       # Low volatility
    
    # Volatility Compression Analysis
    bb_width = (2 * data['close'].rolling(20).std()) / data['close'].rolling(20).mean()
    
    # Range calculations
    daily_range = data['high'] - data['low']
    range_5d_avg = daily_range.rolling(5).mean()
    range_20d_avg = daily_range.rolling(20).mean()
    range_compression = range_5d_avg / range_20d_avg
    
    # Volatility compression
    vol_compression = returns_5d_std / returns_20d_std
    
    # Squeeze Intensity Multiplier
    inverse_bb_width = 1 / (bb_width + 1e-8)
    squeeze_intensity = inverse_bb_width * vol_compression * regime
    
    # Multi-Timeframe Divergence Analysis
    # Calculate divergence between timeframes
    short_medium_div = returns_5d - returns_20d
    medium_long_div = returns_20d - returns_60d
    
    # Triple divergence
    momentum_values = pd.DataFrame({
        'short': returns_5d,
        'medium': returns_20d, 
        'long': returns_60d
    })
    triple_divergence = momentum_values.std(axis=1)
    
    # Weight divergence by momentum strength
    momentum_strength = abs(decay_momentum)
    divergence_magnitude = triple_divergence * momentum_strength
    
    # Breakout Pattern and Volume Confirmation
    # Early Expansion Signal
    range_expansion = (daily_range > range_5d_avg).astype(float)
    volume_expansion = (data['volume'] > volume_5d).astype(float)
    breakout_potential = range_expansion * volume_expansion
    
    # Volatility-Scaled Volume Confirmation
    # True Range calculation
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    atr_14d = true_range.rolling(14).mean()
    volatility_scaled_volume = volume_ratio / (atr_14d + 1e-8)
    
    # Composite Alpha Synthesis
    # Momentum-Volume Convergence Base
    base_signal = decay_momentum
    momentum_volume_multiplier = 1 + volume_momentum_alignment * amount_volume_ratio
    enhanced_base = base_signal * momentum_volume_multiplier
    
    # Divergence and Regime Integration
    divergence_amplifier = 1 + divergence_magnitude * regime
    divergence_enhanced = enhanced_base * divergence_amplifier
    
    # Squeeze-Breakout Enhancement
    squeeze_breakout_multiplier = 1 + (squeeze_intensity * breakout_potential * volatility_scaled_volume)
    final_alpha = divergence_enhanced * squeeze_breakout_multiplier
    
    return final_alpha
