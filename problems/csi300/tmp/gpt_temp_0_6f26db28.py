import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum-adjusted volume divergence and efficiency-weighted trend strength
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Adjusted Volume Divergence
    # Calculate recent volume trend (5-day slope)
    volume_trend = data['volume'].rolling(window=5).apply(
        lambda x: stats.linregress(range(len(x)), x)[0], raw=False
    )
    
    # Calculate price momentum (10-day rate of change)
    momentum = data['close'].pct_change(periods=10)
    
    # Adjust volume by momentum
    momentum_adjusted_volume = volume_trend * np.sign(momentum) * np.abs(momentum)
    
    # High-Low Range Efficiency
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = np.abs(data['high'] - data['close'].shift(1))
    tr3 = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate price efficiency
    price_change = np.abs(data['close'] - data['close'].shift(1))
    price_efficiency = price_change / true_range
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Efficiency-Weighted Trend Strength
    # Calculate price trend slope (20-day)
    trend_slope = data['close'].rolling(window=20).apply(
        lambda x: stats.linregress(range(len(x)), x)[0], raw=False
    )
    
    # Measure noise-to-signal ratio using rolling volatility
    price_volatility = data['close'].pct_change().rolling(window=20).std()
    noise_ratio = price_volatility / np.abs(trend_slope)
    noise_ratio = noise_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Calculate trend quality (inverse of noise ratio)
    trend_quality = 1 / (1 + noise_ratio)
    
    # Weight by market efficiency using volume impact
    # Use amount/volume as proxy for average price impact
    avg_trade_size = data['amount'] / data['volume']
    avg_trade_size_norm = (avg_trade_size - avg_trade_size.rolling(50).mean()) / avg_trade_size.rolling(50).std()
    
    # Market friction factor (higher friction = lower efficiency)
    market_friction = 1 / (1 + np.exp(-avg_trade_size_norm))
    
    # Efficiency-weighted trend strength
    efficiency_weighted_trend = trend_slope * trend_quality * (1 - market_friction)
    
    # Combine factors
    # Normalize both components
    norm_momentum_volume = (momentum_adjusted_volume - momentum_adjusted_volume.rolling(50).mean()) / momentum_adjusted_volume.rolling(50).std()
    norm_efficiency_trend = (efficiency_weighted_trend - efficiency_weighted_trend.rolling(50).mean()) / efficiency_weighted_trend.rolling(50).std()
    
    # Final alpha factor
    alpha_factor = 0.6 * norm_momentum_volume + 0.4 * norm_efficiency_trend
    
    return alpha_factor
