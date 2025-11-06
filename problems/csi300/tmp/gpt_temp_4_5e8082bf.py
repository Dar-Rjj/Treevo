import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Alpha Factor combining entropy, momentum, efficiency, microstructure pressure,
    volatility skewness, and volume surge signals.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Entropy-Momentum Integration
    # Price entropy across different periods
    def calculate_entropy(series, window):
        returns = series.pct_change().dropna()
        if len(returns) < window:
            return pd.Series(index=series.index, dtype=float)
        
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
            
            window_returns = returns.iloc[i-window:i]
            if len(window_returns) < 2:
                entropy_values.append(np.nan)
                continue
            
            # Calculate entropy using histogram-based approach
            hist, _ = np.histogram(window_returns, bins=min(10, len(window_returns)), density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=series.index)
    
    entropy_5d = calculate_entropy(data['close'], 5)
    entropy_10d = calculate_entropy(data['close'], 10)
    entropy_20d = calculate_entropy(data['close'], 20)
    
    # Momentum acceleration
    mom_5d = data['close'].pct_change(5)
    mom_10d = data['close'].pct_change(10)
    mom_20d = data['close'].pct_change(20)
    
    mom_accel_5_10 = mom_5d - mom_10d
    mom_accel_10_20 = mom_10d - mom_20d
    
    # Entropy-weighted momentum signals
    entropy_weighted_momentum = (entropy_5d * mom_accel_5_10 + entropy_10d * mom_accel_10_20) / (entropy_5d + entropy_10d)
    
    # Fractal Efficiency Reversal
    # Movement efficiency
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    movement_efficiency = abs(data['close'] - data['close'].shift(1)) / true_range
    
    # Efficiency trend
    eff_3d = movement_efficiency.rolling(window=3, min_periods=2).mean()
    eff_10d = movement_efficiency.rolling(window=10, min_periods=5).mean()
    efficiency_trend = eff_3d - eff_10d
    
    # Efficiency-adjusted return reversal
    efficiency_adjusted_reversal = data['close'].pct_change(3) * (1 - movement_efficiency.rolling(window=3).mean())
    
    # Microstructure Pressure Momentum
    # Intraday pressure
    intraday_pressure = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Pressure momentum
    pressure_5d_sum = intraday_pressure.rolling(window=5, min_periods=3).sum()
    pressure_momentum = pressure_5d_sum.diff(3)
    
    # Pressure-momentum alignment
    pressure_momentum_alignment = np.sign(pressure_momentum) * np.sign(mom_5d)
    
    # Volatility-Entropy Skewness
    # Return skewness
    returns_20d = data['close'].pct_change(20)
    skewness_20d = returns_20d.rolling(window=20, min_periods=10).skew()
    
    # Daily range utilization
    daily_range_utilization = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Skewness-efficiency regime
    skewness_efficiency_regime = skewness_20d * daily_range_utilization.rolling(window=5).mean()
    
    # Volume-Surge Entropy Divergence
    # Volume surge detection
    volume_1d = data['volume']
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_surge_ratio = volume_1d / volume_5d_avg
    
    # Entropy-confirmed momentum divergence
    entropy_momentum_divergence = (entropy_10d * mom_5d) - (entropy_5d * mom_10d)
    
    # Surge-entropy-momentum triple alignment
    surge_entropy_momentum = volume_surge_ratio * entropy_momentum_divergence * mom_5d
    
    # Combine all components with hierarchical weighting
    factor = (
        0.25 * entropy_weighted_momentum.fillna(0) +
        0.20 * efficiency_adjusted_reversal.fillna(0) +
        0.15 * pressure_momentum_alignment.fillna(0) +
        0.20 * skewness_efficiency_regime.fillna(0) +
        0.20 * surge_entropy_momentum.fillna(0)
    )
    
    # Normalize the factor
    factor_std = factor.rolling(window=20, min_periods=10).std()
    normalized_factor = factor / (factor_std + 1e-8)
    
    return normalized_factor
