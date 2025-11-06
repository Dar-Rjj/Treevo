import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Momentum Acceleration factor that combines multi-timeframe momentum
    with volume trend analysis and volatility normalization.
    """
    # Multi-timeframe Momentum Calculation
    # Short-term momentum (1-day)
    short_momentum = (df['close'] / df['close'].shift(1) - 1)  # Close-to-close return
    intraday_momentum = (df['close'] / df['open'] - 1)  # Intraday momentum
    
    # Medium-term momentum (3-day)
    medium_momentum = (df['close'] / df['close'].shift(3) - 1)  # Close-to-close return
    short_acceleration = short_momentum - short_momentum.shift(2)  # Acceleration from short-term
    
    # Long-term momentum (5-day)
    long_momentum = (df['close'] / df['close'].shift(5) - 1)  # Close-to-close return
    medium_acceleration = medium_momentum - medium_momentum.shift(2)  # Acceleration from medium-term
    
    # Volume Trend Analysis
    volume_momentum = (df['volume'] / df['volume'].shift(1) - 1)  # Volume change
    volume_acceleration = volume_momentum - volume_momentum.shift(2)  # Volume acceleration
    
    # Volume-price divergence
    price_volume_divergence = short_momentum - volume_momentum  # Compare price vs volume momentum
    
    # Volume persistence
    volume_trend = df['volume'].rolling(window=3).apply(
        lambda x: 1 if (x.diff().dropna() > 0).all() else (-1 if (x.diff().dropna() < 0).all() else 0),
        raw=False
    )
    
    # Volatility Normalization
    daily_range = (df['high'] - df['low']) / df['close']  # High-low range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1) / df['close']
    
    volatility = (daily_range + true_range) / 2  # Average volatility measure
    
    # Normalize momentum signals by volatility
    short_momentum_norm = short_momentum / volatility.replace(0, np.nan)
    medium_momentum_norm = medium_momentum / volatility.replace(0, np.nan)
    long_momentum_norm = long_momentum / volatility.replace(0, np.nan)
    
    # Signal Combination
    # Weighted combination with higher weights for accelerating momentum
    momentum_weights = pd.DataFrame({
        'short': short_momentum_norm * (1 + abs(short_acceleration)),
        'medium': medium_momentum_norm * (1 + abs(medium_acceleration)),
        'long': long_momentum_norm * (1 + abs(medium_acceleration))
    })
    
    # Volume-confirmed signals get priority
    volume_confirmation = np.where(
        (volume_momentum > 0) & (volume_acceleration > 0), 1.2,
        np.where((volume_momentum < 0) & (volume_acceleration < 0), 0.8, 1.0)
    )
    
    # Divergence pattern recognition
    divergence_score = np.where(
        (short_momentum > 0) & (volume_momentum < 0), -1,
        np.where((short_momentum < 0) & (volume_momentum > 0), -1, 1)
    )
    
    # Combine all components
    weighted_momentum = (
        momentum_weights['short'] * 0.4 + 
        momentum_weights['medium'] * 0.35 + 
        momentum_weights['long'] * 0.25
    )
    
    # Final factor calculation
    factor = (
        weighted_momentum * 
        volume_confirmation * 
        divergence_score * 
        (1 + volume_trend * 0.1)
    )
    
    return factor
