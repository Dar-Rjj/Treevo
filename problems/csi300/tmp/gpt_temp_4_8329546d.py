import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Volume Momentum Divergence
    # Calculate 5-day Volume Moving Average
    vol_ma_5 = data['volume'].rolling(window=5, min_periods=5).mean()
    
    # Calculate 20-day Volume Moving Average
    vol_ma_20 = data['volume'].rolling(window=20, min_periods=20).mean()
    
    # Calculate Volume Momentum Ratio
    vol_momentum_ratio = (vol_ma_5 / vol_ma_20) - 1
    
    # Compute Price Momentum Component
    # Calculate 10-day Price Return
    price_return_10 = (data['close'] / data['close'].shift(9)) - 1
    
    # Calculate 10-day Average True Range
    # First calculate True Range for each day
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_10 = true_range.rolling(window=10, min_periods=10).mean()
    
    # Identify Volatility Regime
    # Calculate 20-day Price Volatility (standard deviation of returns)
    returns = data['close'].pct_change()
    vol_20 = returns.rolling(window=20, min_periods=20).std()
    
    # Calculate median volatility over past 50 days
    vol_median_50 = vol_20.rolling(window=50, min_periods=50).median()
    
    # Classify Volatility Regime
    high_vol_regime = vol_20 > vol_median_50
    
    # Combine Components with Regime Adjustment
    # Calculate Raw Momentum Signal
    raw_signal = (vol_momentum_ratio * price_return_10) / atr_10
    
    # Apply Volatility Regime Scaling
    regime_adjusted_signal = raw_signal.copy()
    regime_adjusted_signal[high_vol_regime] = regime_adjusted_signal[high_vol_regime] * 1.5
    
    # Apply Volume-Weighted Filter
    # Calculate Volume Percentile over past 50 days
    volume_percentile = data['volume'].rolling(window=50, min_periods=50).apply(
        lambda x: (x[-1] > x[:-1]).sum() / len(x[:-1]) if len(x) == 50 else np.nan
    )
    
    # Multiply final factor by Volume Percentile
    final_factor = regime_adjusted_signal * volume_percentile
    
    return final_factor
