import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Momentum Acceleration
    # Price Momentum
    price_change_3d = data['close'] / data['close'].shift(2) - 1
    price_change_6d = data['close'] / data['close'].shift(5) - 1
    
    # Volume Momentum
    volume_change_3d = data['volume'] / data['volume'].shift(2) - 1
    volume_change_6d = data['volume'] / data['volume'].shift(5) - 1
    
    # Compute Convergence Signals
    # Momentum Alignment
    momentum_align_3d = price_change_3d * volume_change_3d
    momentum_align_6d = price_change_6d * volume_change_6d
    
    # Acceleration Divergence
    # Avoid division by zero
    acceleration_divergence = momentum_align_3d / np.where(momentum_align_6d != 0, momentum_align_6d, np.nan)
    acceleration_divergence = np.tanh(acceleration_divergence)
    
    # Incorporate Price Efficiency
    # Intraday Efficiency
    efficiency_ratio = (data['close'] - data['open']) / np.where((data['high'] - data['low']) != 0, (data['high'] - data['low']), np.nan)
    
    # 3-day Efficiency Trend using linear regression
    efficiency_trend = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            window_data = efficiency_ratio.iloc[i-2:i+1]
            if not window_data.isna().any():
                x = np.arange(len(window_data))
                slope, _, _, _, _ = linregress(x, window_data.values)
                efficiency_trend.iloc[i] = slope
            else:
                efficiency_trend.iloc[i] = 0
        else:
            efficiency_trend.iloc[i] = 0
    
    # Volume Concentration
    # Large Trade Indicator
    avg_trade_size = np.where(data['volume'] != 0, data['amount'] / data['volume'], 0)
    
    # 3-day Trade Size Trend using linear regression
    trade_size_trend = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            window_data = avg_trade_size[i-2:i+1]
            if not np.isnan(window_data).any():
                x = np.arange(len(window_data))
                slope, _, _, _, _ = linregress(x, window_data)
                trade_size_trend.iloc[i] = slope
            else:
                trade_size_trend.iloc[i] = 0
        else:
            trade_size_trend.iloc[i] = 0
    
    # Generate Composite Factor
    # Combine Convergence and Efficiency
    composite = acceleration_divergence * efficiency_trend * trade_size_trend
    
    # Apply Dynamic Scaling
    # Recent Volatility (5-day standard deviation of daily ranges)
    daily_ranges = (data['high'] - data['low']).rolling(window=5, min_periods=3).std()
    
    # Scale Factor by Volatility Reciprocal
    volatility_scaling = np.where(daily_ranges != 0, 1 / daily_ranges, 1)
    scaled_composite = composite * volatility_scaling
    
    # Final Factor Adjustment
    # Apply cubic root transformation preserving original sign
    final_factor = np.sign(acceleration_divergence) * np.cbrt(np.abs(scaled_composite))
    
    return final_factor
