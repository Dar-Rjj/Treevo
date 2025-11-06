import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Analysis
    # Compute 5-day log return
    returns_5d = np.log(df['close'] / df['close'].shift(5))
    
    # Apply exponential decay to returns (5-day window, more weight to recent)
    decay_weights = np.exp(-np.arange(5) / 2.5)  # Exponential decay
    decay_weights = decay_weights / decay_weights.sum()
    
    # Create decay-weighted momentum signal
    momentum_signal = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_returns = returns_5d.iloc[i-4:i+1]  # Current + last 4 days
        momentum_signal.iloc[i] = (window_returns * decay_weights).sum()
    
    # Reversal Detection
    # Identify price extremes using recent high-low range
    high_5d = df['high'].rolling(window=5, min_periods=1).max()
    low_5d = df['low'].rolling(window=5, min_periods=1).min()
    range_mid = (high_5d + low_5d) / 2
    price_extreme = (df['close'] - range_mid) / (high_5d - low_5d + 1e-8)
    
    # Assess momentum exhaustion
    momentum_ma = momentum_signal.rolling(window=10, min_periods=1).mean()
    momentum_std = momentum_signal.rolling(window=10, min_periods=1).std()
    momentum_zscore = (momentum_signal - momentum_ma) / (momentum_std + 1e-8)
    
    # Detect overbought/oversold conditions
    overbought = (price_extreme > 0.7) & (momentum_zscore > 1.5)
    oversold = (price_extreme < -0.7) & (momentum_zscore < -1.5)
    
    # Volume Confirmation with Trend Analysis
    # Compute volume change
    volume_change = df['volume'] / (df['volume'].shift(1) + 1e-8) - 1
    
    # Apply decay weights to volume changes (3-day window)
    vol_decay_weights = np.exp(-np.arange(3) / 1.5)
    vol_decay_weights = vol_decay_weights / vol_decay_weights.sum()
    
    # Generate volume trend confirmation signal
    volume_trend = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        window_vol = volume_change.iloc[i-2:i+1]  # Current + last 2 days
        volume_trend.iloc[i] = (window_vol * vol_decay_weights).sum()
    
    # Volume breakout detection
    vol_ma = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_breakout = df['volume'] > (vol_ma * 1.2)
    
    # Signal Synthesis
    # Momentum reversal validation
    reversal_signal = pd.Series(0, index=df.index)
    
    # Bullish reversal: oversold with positive volume confirmation
    bullish_condition = oversold & (volume_trend > 0.1) & volume_breakout
    reversal_signal[bullish_condition] = 1
    
    # Bearish reversal: overbought with negative volume confirmation  
    bearish_condition = overbought & (volume_trend < -0.1) & volume_breakout
    reversal_signal[bearish_condition] = -1
    
    # Combine with momentum for final alpha factor
    alpha_factor = reversal_signal * np.abs(momentum_zscore)
    
    # Fill NaN values
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
