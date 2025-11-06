import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Parameters
    n_atr = 14
    n_acceleration = 5
    n_volume = 10
    regime_percentile = 70
    
    # Volatility Regime Detection
    # Calculate ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=n_atr).mean()
    
    # Classify regime by ATR percentiles
    atr_percentile = atr.rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, regime_percentile)), raw=False)
    high_vol_regime = (atr_percentile == 1)
    
    # Price Acceleration Analysis
    # Compute first derivative (velocity)
    velocity = data['close'].diff()
    
    # Compute second derivative (acceleration)
    acceleration = velocity.diff(n_acceleration)
    
    # Calculate volatility-adjusted acceleration
    vol_adjusted_acceleration = acceleration / atr
    
    # Volume Confirmation
    # Compute volume momentum
    volume_momentum = data['volume'] / data['volume'].rolling(window=n_volume).mean()
    
    # Calculate volume-weighted acceleration score
    volume_weighted_acceleration = vol_adjusted_acceleration * volume_momentum
    
    # Regime-Adaptive Signal
    # High volatility: focus on extreme volatility-adjusted reversals
    high_vol_signal = -volume_weighted_acceleration.rolling(window=5).apply(
        lambda x: x.iloc[-1] if abs(x.iloc[-1]) > np.percentile(np.abs(x), 80) else 0, raw=False
    )
    
    # Low volatility: emphasize acceleration divergence patterns
    low_vol_signal = volume_weighted_acceleration.rolling(window=10).apply(
        lambda x: x.iloc[-1] - np.mean(x), raw=False
    )
    
    # Combine signals based on regime
    final_signal = pd.Series(np.where(high_vol_regime, high_vol_signal, low_vol_signal), 
                            index=data.index)
    
    # Normalize the final signal
    final_signal = (final_signal - final_signal.rolling(window=20).mean()) / final_signal.rolling(window=20).std()
    
    return final_signal
