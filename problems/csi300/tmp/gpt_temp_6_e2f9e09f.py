import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Micro-Structure Momentum Divergence factor
    Combines price and volume fractal dynamics with micro-structure efficiency analysis
    """
    data = df.copy()
    
    # Helper function for fractal dimension approximation using Hurst exponent
    def hurst_exponent(series, window):
        """Calculate Hurst exponent as proxy for fractal dimension"""
        lags = range(2, min(8, window))
        tau = []
        for lag in lags:
            if len(series) >= lag:
                tau.append(np.std(np.subtract(series[lag:].values, series[:-lag].values)))
            else:
                tau.append(np.nan)
        
        if len(tau) > 1 and not np.any(np.isnan(tau)):
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return poly[0]  # Hurst exponent
        else:
            return np.nan
    
    # Price fractal momentum components
    def calculate_price_fractal_momentum():
        """Multi-scale price fractal momentum analysis"""
        # 3-day price fractal dimension
        price_fractal_3d = data['close'].rolling(window=10, min_periods=5).apply(
            lambda x: hurst_exponent(x, 10), raw=False
        )
        
        # 5-day price fractal dimension  
        price_fractal_5d = data['close'].rolling(window=15, min_periods=8).apply(
            lambda x: hurst_exponent(x, 15), raw=False
        )
        
        # Fractal momentum divergence
        fractal_momentum_3d = price_fractal_3d.diff(3)
        fractal_momentum_5d = price_fractal_5d.diff(5)
        
        # Short vs medium-term divergence
        fractal_divergence = fractal_momentum_3d - fractal_momentum_5d.rolling(window=3).mean()
        
        return fractal_divergence, fractal_momentum_3d
    
    # Volume fractal dynamics
    def calculate_volume_fractal_dynamics():
        """Volume pattern complexity analysis"""
        # Volume fractal dimension
        volume_fractal = data['volume'].rolling(window=10, min_periods=5).apply(
            lambda x: hurst_exponent(x, 10), raw=False
        )
        
        # Volume fractal momentum
        volume_fractal_momentum = volume_fractal.diff(3)
        
        # Volume complexity acceleration
        volume_acceleration = volume_fractal_momentum.diff(2)
        
        # Volume fractal regimes
        volume_complexity_avg = volume_fractal.rolling(window=10).mean()
        volume_regime = volume_fractal - volume_complexity_avg
        
        return volume_fractal_momentum, volume_acceleration, volume_regime
    
    # Micro-structure efficiency analysis
    def calculate_micro_efficiency():
        """Price-volume efficiency and micro-flow analysis"""
        # Price efficiency ratio: (Close - Open) / (High - Low)
        efficiency_ratio = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        
        # Efficiency momentum
        efficiency_momentum = efficiency_ratio.rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )
        
        # Price momentum vs efficiency divergence
        price_momentum = data['close'].pct_change(3)
        micro_flow_divergence = price_momentum - efficiency_momentum
        
        return efficiency_ratio, efficiency_momentum, micro_flow_divergence
    
    # Volatility fractal analysis
    def calculate_volatility_fractal():
        """True range complexity and volume-volatility relationship"""
        # True range
        true_range = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Volatility fractal dimension
        vol_fractal = true_range.rolling(window=10, min_periods=5).apply(
            lambda x: hurst_exponent(x, 10), raw=False
        )
        
        # Volume-volatility fractal relationship
        volume_fractal = data['volume'].rolling(window=10, min_periods=5).apply(
            lambda x: hurst_exponent(x, 10), raw=False
        )
        vol_vol_relationship = volume_fractal * vol_fractal
        
        return vol_fractal, vol_vol_relationship
    
    # Calculate all components
    price_fractal_div, price_momentum_3d = calculate_price_fractal_momentum()
    vol_fractal_momentum, vol_acceleration, vol_regime = calculate_volume_fractal_dynamics()
    efficiency_ratio, efficiency_momentum, micro_flow_div = calculate_micro_efficiency()
    vol_fractal, vol_vol_rel = calculate_volatility_fractal()
    
    # Adaptive fractal signal generation
    # Compression detection (low fractal dimension)
    price_fractal_3d = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: hurst_exponent(x, 10), raw=False
    )
    compression_signal = (price_fractal_3d < price_fractal_3d.rolling(window=20).quantile(0.3)).astype(int)
    
    # Expansion signals (fractal dimension breakouts)
    expansion_signal = (price_fractal_3d > price_fractal_3d.rolling(window=20).quantile(0.7)).astype(int)
    
    # Normal regime (middle fractal complexity)
    normal_regime = ((price_fractal_3d >= price_fractal_3d.rolling(window=20).quantile(0.3)) & 
                    (price_fractal_3d <= price_fractal_3d.rolling(window=20).quantile(0.7))).astype(int)
    
    # High complexity regime
    high_complexity = (price_fractal_3d > price_fractal_3d.rolling(window=20).quantile(0.8)).astype(int)
    
    # Generate final factor with regime weighting
    # Compression regime: focus on expansion anticipation
    compression_factor = (
        price_fractal_div * 1.5 +  # Enhanced sensitivity to divergence
        vol_fractal_momentum * 0.8 +  # Volume confirmation
        micro_flow_div * 1.2  # Micro-flow signals
    ) * compression_signal
    
    # Normal regime: balanced approach
    normal_factor = (
        price_fractal_div * 1.0 +
        vol_fractal_momentum * 1.0 +
        efficiency_momentum * 0.7 +
        vol_vol_rel * 0.5
    ) * normal_regime
    
    # High complexity regime: mean reversion focus
    high_complexity_factor = (
        -price_fractal_div * 1.3 +  # Contrarian signal
        -vol_acceleration * 0.8 +  # Fade volume acceleration
        efficiency_momentum * 0.6  # Efficiency normalization
    ) * high_complexity
    
    # Combine all regimes
    final_factor = (
        compression_factor + 
        normal_factor + 
        high_complexity_factor
    )
    
    # Smooth the final factor
    final_factor_smoothed = final_factor.rolling(window=3, min_periods=1).mean()
    
    return final_factor_smoothed
