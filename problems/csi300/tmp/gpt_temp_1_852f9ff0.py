import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration
    # 3-day Price Momentum
    mom_3d = data['close'] / data['close'].shift(3) - 1
    
    # 8-day Price Momentum
    mom_8d = data['close'] / data['close'].shift(8) - 1
    
    # Acceleration Signal
    accel_signal = mom_3d - mom_8d
    
    # Calculate Volume Convergence
    # 3-day Volume Momentum
    vol_mom_3d = data['volume'] / data['volume'].shift(3) - 1
    
    # 8-day Volume Momentum
    vol_mom_8d = data['volume'] / data['volume'].shift(8) - 1
    
    # Volume Convergence Ratio (with epsilon to avoid division by zero)
    epsilon = 1e-8
    vol_convergence_ratio = vol_mom_3d / (vol_mom_8d + epsilon)
    
    # Calculate Price-Volume Alignment
    # Daily percentage changes
    price_returns = data['close'].pct_change()
    volume_changes = data['volume'].pct_change()
    
    # 10-day rolling correlation
    corr_window = 10
    price_volume_corr = price_returns.rolling(window=corr_window).corr(volume_changes)
    
    # Alignment Strength
    alignment_strength = price_volume_corr.abs() * np.sign(mom_3d)
    
    # Calculate Decay-Adjusted Momentum
    decay_factor = 0.9
    decay_weights = np.array([decay_factor ** i for i in range(5)])
    decay_weights = decay_weights / decay_weights.sum()  # Normalize
    
    # Calculate weighted momentum using past 5 days of returns
    daily_returns = price_returns
    weighted_momentum = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 4:  # Need at least 5 days of data
            recent_returns = [daily_returns.iloc[i-j] for j in range(5)]
            if not any(pd.isna(recent_returns)):
                weighted_momentum.iloc[i] = np.sum(decay_weights * recent_returns)
    
    # Combine Core Components
    # Multiply Acceleration Signal by Volume Convergence Ratio
    combined_signal = accel_signal * vol_convergence_ratio
    
    # Apply Price-Volume Alignment Filter
    alignment_multiplier = pd.Series(1.0, index=data.index)
    alignment_multiplier[price_volume_corr > 0.3] = 1.2
    alignment_multiplier[price_volume_corr < -0.3] = 0.8
    
    combined_signal = combined_signal * alignment_multiplier
    
    # Incorporate Decay-Adjusted Momentum with adaptive scaling
    momentum_scaling = np.abs(combined_signal).rolling(window=5).mean()
    momentum_scaling = momentum_scaling.replace(0, 1)  # Avoid division by zero
    combined_signal = combined_signal + (weighted_momentum / momentum_scaling)
    
    # Apply Direction Consistency Check
    components = [accel_signal, vol_convergence_ratio, weighted_momentum]
    consistency_multiplier = pd.Series(1.0, index=data.index)
    
    for i in range(len(data)):
        if i >= 8:  # Need enough data for all components
            signs = [np.sign(comp.iloc[i]) for comp in components if not pd.isna(comp.iloc[i])]
            if len(signs) == 3:
                if len(set(signs)) == 1:  # All same sign
                    consistency_multiplier.iloc[i] = 1.5
                elif len(set(signs)) == 3:  # All different signs
                    consistency_multiplier.iloc[i] = 0.7
    
    combined_signal = combined_signal * consistency_multiplier
    
    # Apply Dynamic Scaling
    # Calculate Recent Volatility Measure
    # True Range calculation
    high_low = data['high'] - data['low']
    high_close_prev = np.abs(data['high'] - data['close'].shift(1))
    low_close_prev = np.abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    avg_true_range = true_range.rolling(window=5).mean()
    
    # Scale by average price level
    avg_price = data['close'].rolling(window=5).mean()
    volatility_measure = avg_true_range / (avg_price + epsilon)
    
    # Calculate Volume Stability Factor
    volume_cv = data['volume'].rolling(window=5).std() / (data['volume'].rolling(window=5).mean() + epsilon)
    volume_stability = 1 / (volume_cv + epsilon)
    
    # Apply Final Adjustments
    volatility_stability = 1 / (volatility_measure + epsilon)
    volatility_stability = volatility_stability / volatility_stability.rolling(window=10).mean()
    
    # Apply scaling factors
    final_signal = combined_signal * volatility_stability * volume_stability
    
    # Apply gradual decay to extreme values using tanh
    signal_std = final_signal.rolling(window=20).std()
    signal_std = signal_std.replace(0, 1)  # Avoid division by zero
    normalized_signal = final_signal / (signal_std + epsilon)
    bounded_signal = np.tanh(normalized_signal)
    
    return bounded_signal
