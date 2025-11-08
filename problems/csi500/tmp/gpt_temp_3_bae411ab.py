import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Momentum with Dynamic Volume Confirmation
    """
    # Core Momentum Component
    # Price Momentum Calculation
    short_return = df['close'].pct_change(1)
    medium_return = df['close'].pct_change(5)
    long_return = df['close'].pct_change(20)
    
    # Volatility Normalization
    # Intraday Volatility Proxy
    daily_range = (df['high'] - df['low']) / df['close']
    rolling_volatility = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Adaptive Scaling with Inverse Hyperbolic Sine Transform
    def normalize_momentum(returns, volatility):
        normalized = returns / (volatility + 1e-8)
        return np.arcsinh(normalized)
    
    short_momentum = normalize_momentum(short_return, rolling_volatility)
    medium_momentum = normalize_momentum(medium_return, rolling_volatility)
    long_momentum = normalize_momentum(long_return, rolling_volatility)
    
    # Volume Confirmation Layer
    # Volume Trend Analysis
    volume_median = df['volume'].rolling(window=20, min_periods=10).median()
    volume_ratio = df['volume'] / (volume_median + 1e-8)
    
    volume_acceleration = df['volume'].pct_change(1)
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan
    )
    
    # Dynamic Thresholding with Sigmoid Activation
    def sigmoid_activation(x, center=1.0, slope=3.0):
        return 1 / (1 + np.exp(-slope * (x - center)))
    
    volume_confidence = (
        0.4 * sigmoid_activation(volume_ratio, center=1.2, slope=2.0) +
        0.3 * sigmoid_activation(volume_acceleration + 1, center=1.1, slope=4.0) +
        0.3 * sigmoid_activation(np.abs(volume_trend) / (df['volume'].rolling(20).std() + 1e-8), 
                                center=0.5, slope=3.0)
    )
    
    # Factor Integration
    # Multi-timeframe Combination
    combined_momentum = (
        0.4 * short_momentum +
        0.35 * medium_momentum +
        0.25 * long_momentum
    )
    
    # Volume-Weighted Output with Nonlinear Scaling
    raw_factor = combined_momentum * volume_confidence
    final_factor = np.sign(raw_factor) * np.power(np.abs(raw_factor), 1/3)
    
    return final_factor
