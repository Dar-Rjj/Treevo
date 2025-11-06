import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap Efficiency Framework
    # Short-term gap efficiency: (Open[t] - Close[t-1]) / (High[t-2:t] - Low[t-2:t])
    data['gap_short'] = (data['open'] - data['close'].shift(1)) / (
        data['high'].rolling(window=3, min_periods=3).max() - 
        data['low'].rolling(window=3, min_periods=3).min()
    )
    
    # Medium-term gap efficiency: (Open[t] - Close[t-1]) / (High[t-4:t] - Low[t-4:t])
    data['gap_medium'] = (data['open'] - data['close'].shift(1)) / (
        data['high'].rolling(window=5, min_periods=5).max() - 
        data['low'].rolling(window=5, min_periods=5).min()
    )
    
    # Gap efficiency divergence
    data['gap_divergence'] = data['gap_short'] - data['gap_medium']
    
    # Range Efficiency Framework
    # Daily range efficiency
    data['range_daily'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Short-term range efficiency (3-day average)
    data['range_short'] = data['range_daily'].rolling(window=3, min_periods=3).mean()
    
    # Medium-term range efficiency (8-day average)
    data['range_medium'] = data['range_daily'].rolling(window=8, min_periods=8).mean()
    
    # Range efficiency divergence
    data['range_divergence'] = data['range_short'] - data['range_medium']
    
    # Volatility-Weighted Momentum
    # True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 5-day Average True Range
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    
    # Daily momentum
    epsilon = 1e-8
    data['daily_momentum'] = (data['close'] - data['close'].shift(1)) * data['amount'] / (data['true_range'] + epsilon)
    
    # Volatility-weighted momentum
    data['vol_weighted_momentum'] = data['daily_momentum'] / (data['atr_5'] + epsilon)
    
    # Volume Confirmation
    # Volume Spike: Volume[t] > Median(Volume[t-4:t])
    volume_median = data['volume'].rolling(window=5, min_periods=5).median()
    data['volume_spike'] = (data['volume'] > volume_median).astype(int)
    
    # Consecutive Spike Count
    data['spike_count'] = 0
    for i in range(1, len(data)):
        if data['volume_spike'].iloc[i] == 1:
            data['spike_count'].iloc[i] = data['spike_count'].iloc[i-1] + 1
        else:
            data['spike_count'].iloc[i] = 0
    
    # Volume Persistence Weight
    data['volume_persistence'] = data['spike_count'] / 5.0
    
    # Gap-Range Alignment
    # Gap pattern
    data['gap_pattern'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Range flow
    data['range_flow'] = np.sign(data['amount'] * (data['close'] - data['open']) / (data['true_range'] + epsilon))
    
    # Factor Integration
    # Cross-efficiency divergence
    data['cross_efficiency_div'] = data['gap_divergence'] * data['range_divergence']
    
    # Volume-enhanced momentum
    data['volume_enhanced_momentum'] = data['vol_weighted_momentum'] * data['volume_persistence']
    
    # Alignment context
    data['alignment_context'] = data['gap_pattern'] * data['range_flow']
    
    # Final Factor
    data['factor'] = data['cross_efficiency_div'] * data['volume_enhanced_momentum'] * data['alignment_context']
    
    return data['factor']
