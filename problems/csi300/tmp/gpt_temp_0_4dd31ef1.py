import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum with Volatility Stabilization
    # Dual-Timeframe Price Momentum
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Volatility-Adjust Both Momentum Terms
    data['daily_range'] = data['high'] - data['low']
    data['range_volatility'] = data['daily_range'].rolling(window=10).std()
    
    # Avoid division by zero
    range_vol_adj = data['range_volatility'].replace(0, np.nan)
    data['vol_adj_momentum_5d'] = data['momentum_5d'] / range_vol_adj
    data['vol_adj_momentum_10d'] = data['momentum_10d'] / range_vol_adj
    
    # Volume Dynamics Integration
    # Volume Stability Assessment
    data['volume_std_10d'] = data['volume'].rolling(window=10).std()
    data['volume_avg_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_stability_ratio'] = data['volume_std_10d'] / data['volume_avg_10d'].replace(0, np.nan)
    
    # Volume Trend Divergence
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma_5d'].replace(0, np.nan)
    
    # Volume-Price Alignment
    data['volume_to_price_ratio'] = data['volume'] / data['close'].replace(0, np.nan)
    data['volume_change'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan) - 1
    
    # Range Momentum Component
    data['range_momentum'] = (data['daily_range'] - data['daily_range'].shift(5)) / data['daily_range'].shift(5).replace(0, np.nan)
    data['raw_range_diff'] = data['daily_range'] - data['daily_range'].shift(5)
    
    # Final Alpha Synthesis
    # Combine Volatility-Adjusted Momentum Terms
    vol_adj_momentum_avg = (data['vol_adj_momentum_5d'] + data['vol_adj_momentum_10d']) / 2
    momentum_component = vol_adj_momentum_avg * data['volume_stability_ratio']
    
    # Incorporate Volume Dynamics
    volume_component = momentum_component * data['volume_momentum'] * data['volume_to_price_ratio']
    
    # Integrate Range Dynamics
    range_component = volume_component * data['range_momentum']
    
    # Apply Direction Adjustment
    # Calculate Price Trend using Close price slope over past 5 days
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window and not np.any(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['price_trend'] = calculate_slope(data['close'], 5)
    data['volume_trend'] = calculate_slope(data['volume'], 5)
    
    # Calculate direction adjustment
    direction_adjustment = np.sign(data['price_trend'] * data['volume_trend'])
    direction_adjustment = direction_adjustment.replace(0, 1)  # Default to positive if zero
    
    # Final alpha factor
    alpha = range_component * direction_adjustment
    
    # Scale by 1000 for better numerical properties
    alpha = alpha * 1000
    
    return alpha
