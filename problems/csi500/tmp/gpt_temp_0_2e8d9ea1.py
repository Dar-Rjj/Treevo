import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Acceleration with Volume Divergence factor
    Combines multi-timeframe momentum acceleration with volume anomaly detection
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate multi-timeframe momentum acceleration
    # Short-term momentum (3-day ROC)
    short_momentum = data['close'].pct_change(periods=3)
    
    # Medium-term momentum (5-day ROC)
    medium_momentum = data['close'].pct_change(periods=5)
    
    # Acceleration signal (short-term minus medium-term momentum)
    acceleration = short_momentum - medium_momentum
    
    # Detect volume anomalies using Z-score
    volume_mean = data['volume'].rolling(window=5).mean()
    volume_std = data['volume'].rolling(window=5).std()
    volume_zscore = (data['volume'] - volume_mean) / volume_std
    
    # Assess volatility environment using high-low range
    daily_range = (data['high'] - data['low']) / data['close']
    volatility_regime = daily_range.rolling(window=5).std()
    
    # Generate combined alpha signal with regime-aware weighting
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 5:  # Skip initial periods for rolling calculations
            alpha_signal.iloc[i] = 0
            continue
            
        # Regime-aware weighting - dampen in volatile conditions
        volatility_weight = 1.0 / (1.0 + volatility_regime.iloc[i])
        
        # Volume-confirmed momentum signals
        if acceleration.iloc[i] > 0.02:  # Strong acceleration
            if volume_zscore.iloc[i] > 1.5:  # High volume divergence
                # Strong acceleration + high volume: Strong continuation
                signal_strength = acceleration.iloc[i] * (1 + volume_zscore.iloc[i] * 0.1)
            else:  # Low volume
                # Strong acceleration + low volume: Weak signal
                signal_strength = acceleration.iloc[i] * 0.5
        elif acceleration.iloc[i] < -0.02:  # Deceleration
            if volume_zscore.iloc[i] > 1.5:  # High volume divergence
                # Deceleration + high volume: Potential reversal
                signal_strength = acceleration.iloc[i] * (1 + volume_zscore.iloc[i] * 0.2)
            else:  # Low volume
                # Deceleration + low volume: Noise
                signal_strength = acceleration.iloc[i] * 0.3
        else:  # Neutral acceleration
            signal_strength = acceleration.iloc[i] * 0.1
        
        # Apply regime weighting
        alpha_signal.iloc[i] = signal_strength * volatility_weight
    
    # Normalize the final signal
    alpha_signal = (alpha_signal - alpha_signal.mean()) / alpha_signal.std()
    
    return alpha_signal
