import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum & Volume Components
    # 1-day price momentum
    data['momentum_1d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # 5-day price momentum
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # 1-day volume change
    data['volume_change_1d'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    
    # Volume persistence over last 5 days
    volume_increase_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            window = data['volume'].iloc[i-4:i+1]  # t-4 to t
            prev_window = data['volume'].iloc[i-5:i]  # t-5 to t-1
            count = sum(window.iloc[j] > prev_window.iloc[j] for j in range(5))
            volume_increase_count.iloc[i] = count / 5.0  # Normalize to 0-1
    
    # Multi-Timeframe Integration
    # Volume-weighted price momentum
    data['vw_momentum_1d'] = (data['close'] - data['close'].shift(1)) * data['volume']
    
    # Cumulative volume-weighted return (5-day window)
    data['cum_vw_return_5d'] = 0.0
    for i in range(len(data)):
        if i >= 5:
            cum_sum = 0
            for j in range(i-4, i+1):  # t-4 to t
                if j > 0:
                    price_diff = data['close'].iloc[j] - data['close'].iloc[j-1]
                    cum_sum += price_diff * data['volume'].iloc[j]
            data['cum_vw_return_5d'].iloc[i] = cum_sum
    
    # Momentum divergence
    data['momentum_divergence'] = data['momentum_1d'] - data['momentum_5d']
    
    # Volume Concentration & Spike Analysis
    # Since we don't have intraday data, we'll use proxies
    # Opening volume concentration proxy: first hour = first 25% of daily volume
    data['opening_volume_ratio'] = data['volume'] * 0.25 / data['volume']  # Proxy = 0.25
    
    # Closing volume concentration proxy: last hour = last 25% of daily volume  
    data['closing_volume_ratio'] = data['volume'] * 0.25 / data['volume']  # Proxy = 0.25
    
    # Volume spike detection
    data['volume_spike_ratio'] = 0.0
    for i in range(len(data)):
        if i >= 5:
            avg_volume = data['volume'].iloc[i-4:i+1].mean()  # t-4 to t
            data['volume_spike_ratio'].iloc[i] = data['volume'].iloc[i] / avg_volume
    
    # Divergence Signal Processing
    # Volume-weighted momentum divergence
    data['vw_momentum_divergence'] = data['momentum_divergence'] * data['volume']
    
    # Opening volume momentum
    data['opening_volume_momentum'] = data['opening_volume_ratio'] * data['momentum_1d']
    
    # Final Factor: Volume-Weighted Momentum Divergence × Volume Persistence
    # Use volume persistence from earlier calculation
    data['final_factor'] = data['vw_momentum_divergence'] * volume_increase_count
    
    # Handle NaN values
    data['final_factor'] = data['final_factor'].fillna(0)
    
    return data['final_factor']
