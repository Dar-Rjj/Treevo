import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Calculation
    close = df['close']
    
    # Short-term momentum (5-day)
    short_momentum = (close / close.shift(5) - 1)
    short_momentum_ewm = short_momentum.ewm(alpha=0.2, adjust=False).mean()  # decay=0.8
    
    # Medium-term momentum (15-day)
    medium_momentum = (close / close.shift(15) - 1)
    medium_momentum_ewm = medium_momentum.ewm(alpha=0.1, adjust=False).mean()  # decay=0.9
    
    # Long-term momentum (30-day)
    long_momentum = (close / close.shift(30) - 1)
    long_momentum_ewm = long_momentum.ewm(alpha=0.05, adjust=False).mean()  # decay=0.95
    
    # Volume-Enhanced Price Signal Generation
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Daily price range signal
    normalized_range = (high - low) / close
    range_signal = normalized_range * np.log(volume + 1)
    
    # Volume-weighted returns
    daily_returns = close.pct_change()
    weighted_returns = daily_returns * np.cbrt(volume)
    
    # Combined volume-price indicator
    combined_volume_price = range_signal + weighted_returns
    volume_price_indicator = combined_volume_price.ewm(span=8, adjust=False).mean()
    
    # Momentum-Volume Convergence Measurement
    # Rolling correlations
    corr_short = short_momentum_ewm.rolling(window=10).corr(volume_price_indicator)
    corr_medium = medium_momentum_ewm.rolling(window=20).corr(volume_price_indicator)
    corr_long = long_momentum_ewm.rolling(window=5).corr(volume_price_indicator)
    
    # Convergence strength
    weighted_corr = 0.5 * corr_short + 0.3 * corr_medium + 0.2 * corr_long
    convergence_std = weighted_corr.rolling(window=8).std()
    convergence_strength = weighted_corr * convergence_std
    
    # Adaptive decay to convergence signal
    recent_volatility = daily_returns.rolling(window=10).std()
    decay_rate = np.where(recent_volatility > recent_volatility.median(), 0.15, 0.05)
    adaptive_convergence = convergence_strength.ewm(alpha=decay_rate, adjust=False).mean()
    
    # Final Alpha Factor Construction
    # Combine momentum signals
    combined_momentum = (0.4 * short_momentum_ewm + 
                        0.35 * medium_momentum_ewm + 
                        0.25 * long_momentum_ewm)
    
    momentum_acceleration = short_momentum_ewm - medium_momentum_ewm
    enhanced_momentum = combined_momentum + 0.2 * momentum_acceleration
    
    # Integrate with convergence measure
    momentum_convergence = enhanced_momentum * adaptive_convergence
    volume_scaled = momentum_convergence * np.log(volume + 1)
    
    # Apply volatility-aware smoothing
    volatility = daily_returns.rolling(window=15).std()
    inverse_volatility = 1 / (volatility + 1e-8)  # Avoid division by zero
    
    raw_factor = volume_scaled * inverse_volatility
    final_factor = raw_factor.ewm(alpha=0.1, adjust=False).mean()  # decay=0.9
    
    return final_factor
