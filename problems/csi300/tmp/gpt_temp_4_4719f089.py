import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Range Asymmetry Momentum factor
    Combines multi-scale range analysis, price rejection asymmetry, 
    range-momentum divergence, and volume-range confirmation
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Range Analysis
    for window in [5, 10, 20]:
        # Range fractal dimension approximation using Hurst exponent method
        data[f'range_{window}'] = data['high'] - data['low']
        data[f'log_range_{window}'] = np.log(data[f'range_{window}'])
        data[f'hurst_{window}'] = 2 - data[f'log_range_{window}'].rolling(window=window).apply(
            lambda x: np.polyfit(np.log(np.arange(window) + 1), x.values, 1)[0] if len(x) == window else np.nan
        )
        
        # Volume-range correlation persistence
        data[f'vol_range_corr_{window}'] = data['volume'].rolling(window=window).corr(data[f'range_{window}'])
    
    # Price Rejection Asymmetry
    data['upper_rejection'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['lower_rejection'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['rejection_asymmetry'] = data['upper_rejection'] - data['lower_rejection']
    
    # Range-Momentum Divergence
    for window in [5, 10, 20]:
        # Range momentum (range expansion/contraction)
        data[f'range_momentum_{window}'] = data[f'range_{window}'] / data[f'range_{window}'].shift(window)
        
        # Price momentum
        data[f'price_momentum_{window}'] = data['close'] / data['close'].shift(window)
        
        # Range-price momentum divergence
        data[f'momentum_divergence_{window}'] = (
            np.sign(data[f'range_momentum_{window}'] - 1) != np.sign(data[f'price_momentum_{window}'] - 1)
        ).astype(float)
    
    # Volume-Range Confirmation
    # Volume-range fractal alignment (correlation between volume and range Hurst exponents)
    data['vol_hurst_5'] = np.log(data['volume']).rolling(window=5).apply(
        lambda x: np.polyfit(np.log(np.arange(5) + 1), x.values, 1)[0] if len(x) == 5 else np.nan
    )
    data['hurst_alignment'] = data['hurst_5'].rolling(window=10).corr(data['vol_hurst_5'])
    
    # Extreme range-volume event detection
    data['range_zscore'] = (data['range_5'] - data['range_5'].rolling(window=20).mean()) / data['range_5'].rolling(window=20).std()
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['extreme_event'] = ((data['range_zscore'] > 2) & (data['volume_zscore'] > 2)).astype(float)
    
    # Signal Generation
    # Regime detection using expanding/contracting ranges
    data['range_regime'] = data['range_5'] / data['range_5'].rolling(window=20).mean()
    regime_weights = np.where(
        data['range_regime'] > 1.1, 1.2,  # Expanding regime
        np.where(data['range_regime'] < 0.9, 0.8, 1.0)  # Contracting regime
    )
    
    # Combine divergence strength with range persistence
    divergence_strength = (
        data['momentum_divergence_5'] * 0.4 + 
        data['momentum_divergence_10'] * 0.3 + 
        data['momentum_divergence_20'] * 0.3
    )
    
    range_persistence = (
        data['hurst_5'] * 0.5 + 
        data['hurst_10'] * 0.3 + 
        data['hurst_20'] * 0.2
    )
    
    # Scale by volume-range cluster intensity
    volume_range_intensity = (
        data['vol_range_corr_5'] * 0.5 + 
        data['vol_range_corr_10'] * 0.3 + 
        data['vol_range_corr_20'] * 0.2
    )
    
    # Final factor calculation
    factor = (
        data['rejection_asymmetry'] * 
        divergence_strength * 
        range_persistence * 
        (1 + volume_range_intensity) * 
        regime_weights * 
        (1 + data['extreme_event']) * 
        data['hurst_alignment']
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
