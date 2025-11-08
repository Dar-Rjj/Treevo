import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum-Adjusted Volume Breakout Divergence factor
    Combines volume breakout signals with momentum confirmation and divergence detection
    """
    
    # Compute Volume Breakout Signal
    # Detect Abnormal Volume Surge
    volume_mean_20 = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_surge = data['volume'] / volume_mean_20 - 1
    
    # Identify Price Breakout Direction
    recent_high_5 = data['high'].rolling(window=5, min_periods=3).max()
    recent_low_5 = data['low'].rolling(window=5, min_periods=3).min()
    
    upward_breakout = (data['close'] > recent_high_5.shift(1)).astype(int)
    downward_breakout = (data['close'] < recent_low_5.shift(1)).astype(int)
    
    volume_breakout_signal = volume_surge * (upward_breakout - downward_breakout)
    
    # Calculate Momentum Confirmation
    # Compute Price Rate of Change (10-day)
    roc_10 = data['close'].pct_change(periods=10)
    
    # Assess Momentum Strength
    roc_20_mean = roc_10.rolling(window=20, min_periods=10).mean()
    roc_20_std = roc_10.rolling(window=20, min_periods=10).std()
    roc_zscore = (roc_10 - roc_20_mean) / (roc_20_std + 1e-8)
    
    # Determine Momentum Consistency
    roc_sign_consistency = np.sign(roc_10) * roc_10.rolling(window=5, min_periods=3).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 0 else 0
    )
    
    momentum_strength = roc_zscore * roc_sign_consistency
    
    # Generate Divergence Factor
    # Combine Volume and Momentum Signals
    volume_momentum_combined = volume_breakout_signal * momentum_strength
    
    # Scale by Price Volatility (20-day)
    price_volatility = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    scaled_factor = volume_momentum_combined / (price_volatility + 1e-8)
    
    # Adjust for Market Regime using trend persistence
    price_trend = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    trend_persistence = price_trend.rolling(window=5, min_periods=3).std()
    regime_adjustment = 1 / (1 + np.abs(trend_persistence))
    
    # Final alpha factor
    alpha_factor = scaled_factor * regime_adjustment
    
    return alpha_factor
