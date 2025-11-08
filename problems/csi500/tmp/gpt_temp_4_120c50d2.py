import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum-Volume Regime Factor that combines price momentum, volume acceleration,
    volatility scaling, and market regime detection.
    """
    # Extract price and volume data
    close = data['close']
    volume = data['volume']
    
    # Momentum Component
    # Short-term momentum (5-day)
    short_momentum = close.pct_change(periods=5)
    
    # Medium-term momentum (20-day)
    medium_momentum = close.pct_change(periods=20)
    
    # Volume Acceleration Component
    # 5-day volume change
    volume_5d_avg = volume.rolling(window=5).mean()
    volume_change = (volume - volume_5d_avg) / volume_5d_avg
    
    # Volume acceleration (short-term vs medium-term)
    volume_20d_avg = volume.rolling(window=20).mean()
    volume_acceleration = (volume_5d_avg - volume_20d_avg) / volume_20d_avg
    
    # Volatility Scaling
    # 20-day price volatility
    daily_returns = close.pct_change()
    volatility_20d = daily_returns.rolling(window=20).std()
    # Handle zero volatility cases
    volatility_20d = volatility_20d.replace(0, np.nan)
    
    # Scale momentum by volatility
    short_momentum_scaled = short_momentum / volatility_20d
    medium_momentum_scaled = medium_momentum / volatility_20d
    
    # Regime Detection
    # Market regime (50-day trend using SMA slope)
    sma_50 = close.rolling(window=50).mean()
    market_trend = (sma_50 - sma_50.shift(10)) / sma_50.shift(10)
    
    # Volume regime (20-day volume percentile)
    volume_20d_percentile = volume.rolling(window=20).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Multi-Timeframe Confirmation
    # Weight short-term and medium-term signals based on regime
    # Higher weight to medium-term in trending markets, higher weight to short-term in volatile markets
    market_regime_weight = np.where(market_trend.abs() > market_trend.rolling(50).mean(), 
                                  0.7, 0.3)  # More weight to medium-term in strong trends
    
    volume_regime_weight = np.where(volume_20d_percentile > 0.7, 
                                  0.7, 0.3)  # More weight to short-term in high volume periods
    
    # Combine weights
    short_term_weight = 0.4 * (1 - market_regime_weight) + 0.6 * volume_regime_weight
    medium_term_weight = 1 - short_term_weight
    
    # Combined momentum signal
    combined_momentum = (short_term_weight * short_momentum_scaled + 
                        medium_term_weight * medium_momentum_scaled)
    
    # Final Factor Construction
    # Multiply momentum by volume acceleration and apply regime adjustment
    factor = (combined_momentum * volume_acceleration * 
             (1 + 0.5 * np.sign(market_trend) * np.sign(combined_momentum)))
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
