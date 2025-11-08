import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adjusted Momentum Divergence factor
    Combines intraday momentum divergence with volatility regime detection
    """
    data = df.copy()
    
    # Calculate Intraday Momentum Divergence
    # Compute Intraday Momentum using High, Low, Close
    intraday_range = (data['high'] - data['low']) / data['close']
    close_momentum = (data['close'] - data['open']) / data['open']
    high_low_momentum = (data['high'] - data['low']) / data['close']
    intraday_momentum = close_momentum * np.sign(high_low_momentum)
    
    # Calculate Volume Acceleration
    volume_roc = data['volume'].pct_change(periods=3)
    volume_acceleration = volume_roc.rolling(window=5, min_periods=3).mean()
    
    # Combine Momentum and Volume Signals
    momentum_sign = np.sign(intraday_momentum)
    momentum_volume_divergence = momentum_sign * volume_acceleration
    
    # Identify Current Volatility Regime
    # Calculate Rolling Volatility using High-Low range
    daily_range = (data['high'] - data['low']) / data['close']
    rolling_volatility = daily_range.rolling(window=20, min_periods=15).std()
    
    # Classify Regime Boundaries using 30-day volatility percentiles
    vol_percentile = rolling_volatility.rolling(window=30, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Adjust Signal by Regime Conditions
    regime_adjusted_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 30:  # Insufficient data for regime classification
            regime_adjusted_signal.iloc[i] = momentum_volume_divergence.iloc[i]
            continue
            
        current_vol_percentile = vol_percentile.iloc[i]
        current_momentum_div = momentum_volume_divergence.iloc[i]
        
        # Low Volatility Regime Enhancement
        if current_vol_percentile < -0.5:
            # Amplify momentum signals in low volatility
            signal_strength = 1.5
            # Use volume acceleration as confirmation
            volume_confirmation = 1.0 + abs(volume_acceleration.iloc[i])
            regime_adjusted_signal.iloc[i] = current_momentum_div * signal_strength * volume_confirmation
        
        # High Volatility Regime Adjustment  
        elif current_vol_percentile > 0.5:
            # Scale down momentum signals
            signal_strength = 0.7
            # Focus on volume divergence patterns
            volume_weight = 1.2 if abs(volume_acceleration.iloc[i]) > 0.1 else 0.8
            regime_adjusted_signal.iloc[i] = current_momentum_div * signal_strength * volume_weight
        
        # Normal Regime
        else:
            regime_adjusted_signal.iloc[i] = current_momentum_div
    
    # Regime Transition Detection
    vol_regime_change = vol_percentile.diff(periods=3).abs()
    
    # Adjust signal weights during transitions
    for i in range(len(data)):
        if i < 33:  # Need data for transition detection
            continue
            
        current_transition = vol_regime_change.iloc[i]
        
        if current_transition > 0.8:  # Significant regime change detected
            # Reduce signal strength during unstable periods
            transition_factor = 0.6
            # Increase reliance on volume confirmation
            volume_confirmation = 1.0 + abs(volume_acceleration.iloc[i]) * 2
            regime_adjusted_signal.iloc[i] = (
                regime_adjusted_signal.iloc[i] * transition_factor * volume_confirmation
            )
    
    return regime_adjusted_signal
