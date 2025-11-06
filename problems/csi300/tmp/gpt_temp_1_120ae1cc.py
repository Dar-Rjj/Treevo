import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using momentum-decay adjusted volume profile with recursive volatility filtering
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Parameters
    momentum_window = 20
    decay_factor = 0.9
    volatility_window = 10
    volume_lookback = 50
    
    # Compute Price Momentum with Exponential Decay
    close_prices = data['close']
    
    # Calculate n-day momentum
    momentum = close_prices.pct_change(periods=momentum_window)
    
    # Apply exponential decay weighting to recent momentum
    weights = np.array([decay_factor ** i for i in range(momentum_window, 0, -1)])
    weights = weights / weights.sum()
    
    # Create decay-weighted momentum
    decayed_momentum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= momentum_window:
            recent_momentum = momentum.iloc[i-momentum_window+1:i+1].values
            if len(recent_momentum) == momentum_window:
                decayed_momentum.iloc[i] = np.dot(recent_momentum, weights)
    
    # Volume Profile Adjustment
    volume_data = data['volume']
    
    # Calculate volume percentiles over rolling window
    volume_percentile = volume_data.rolling(window=volume_lookback).apply(
        lambda x: (x[-1] > np.percentile(x[:-1], 70)) if len(x) == volume_lookback else np.nan, 
        raw=True
    )
    
    # Adjust momentum by volume regime
    volume_adjusted_momentum = decayed_momentum.copy()
    
    # High volume regime multiplier (percentile > 70)
    high_volume_mask = volume_percentile > 0.7
    volume_adjusted_momentum[high_volume_mask] = volume_adjusted_momentum[high_volume_mask] * 1.2
    
    # Low volume regime dampener (percentile < 30)
    low_volume_mask = volume_percentile < 0.3
    volume_adjusted_momentum[low_volume_mask] = volume_adjusted_momentum[low_volume_mask] * 0.8
    
    # Recursive Volatility Regime Filter
    high_low_range = (data['high'] - data['low']) / data['close']
    
    # Calculate rolling volatility with recursive smoothing
    volatility = high_low_range.rolling(window=volatility_window).std()
    
    # Apply recursive EMA smoothing to volatility
    alpha_vol = 0.3
    smoothed_volatility = pd.Series(index=data.index, dtype=float)
    smoothed_volatility.iloc[0] = volatility.iloc[0] if not pd.isna(volatility.iloc[0]) else 0
    
    for i in range(1, len(data)):
        if not pd.isna(volatility.iloc[i]):
            smoothed_volatility.iloc[i] = (alpha_vol * volatility.iloc[i] + 
                                         (1 - alpha_vol) * smoothed_volatility.iloc[i-1])
        else:
            smoothed_volatility.iloc[i] = smoothed_volatility.iloc[i-1]
    
    # Regime Classification
    vol_percentile = smoothed_volatility.rolling(window=volatility_window*2).apply(
        lambda x: (x[-1] > np.percentile(x[:-1], 70)) if len(x) == volatility_window*2 else np.nan,
        raw=True
    )
    
    # Generate regime-specific factors
    final_factor = volume_adjusted_momentum.copy()
    
    # High volatility: apply mean reversion dampening
    high_vol_mask = vol_percentile > 0.7
    final_factor[high_vol_mask] = final_factor[high_vol_mask] * 0.7
    
    # Low volatility: enhance momentum continuation
    low_vol_mask = vol_percentile < 0.3
    final_factor[low_vol_mask] = final_factor[low_vol_mask] * 1.3
    
    # Fill NaN values with 0
    final_factor = final_factor.fillna(0)
    
    return final_factor
