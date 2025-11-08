import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Adjusted Momentum with Adaptive Volume Confirmation
    
    Parameters:
    data: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    pandas Series with factor values indexed by date
    """
    
    # Calculate base volatility components
    # Daily Range Ratio: (High[t] - Low[t]) / Close[t-1]
    daily_range_ratio = (data['high'] - data['low']) / data['close'].shift(1)
    
    # 5-day Rolling Standard Deviation of Range Ratio
    volatility = daily_range_ratio.rolling(window=5, min_periods=3).std()
    
    # Compute Volatility-Normalized Returns
    # Add small constant to avoid division by zero
    vol_adj = volatility + 0.0001
    
    # 3-day normalized return
    ret_3d = (data['close'] / data['close'].shift(3) - 1) / vol_adj
    
    # 8-day normalized return  
    ret_8d = (data['close'] / data['close'].shift(8) - 1) / vol_adj
    
    # 20-day normalized return
    ret_20d = (data['close'] / data['close'].shift(20) - 1) / vol_adj
    
    # Adaptive Volume Regime Detection
    # 20-day Rolling Volume Percentile using data up to day t
    def rolling_percentile(series, window):
        return series.rolling(window=window, min_periods=10).apply(
            lambda x: (x.rank(pct=True).iloc[-1] * 100), raw=False
        )
    
    volume_percentile = rolling_percentile(data['volume'], 20)
    
    # Transform to Confidence Multiplier with cubic transformation
    # Map to [0.2, 1.0] range
    volume_confidence = 0.2 + 0.8 * ((volume_percentile / 100) ** 3)
    
    # Multiplicative Signal Combination
    # Composite Momentum Calculation
    momentum_product = ret_3d * ret_8d * ret_20d
    
    # Cube Root Stabilization: sign(product) × abs(product)^(1/3)
    composite_momentum = np.sign(momentum_product) * (np.abs(momentum_product) ** (1/3))
    
    # Volume-Weighted Final Factor
    final_factor = composite_momentum * volume_confidence
    
    return final_factor
