import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-scale Gap Momentum
    data['gap_short'] = (data['open'] - data['close'].shift(1)) / (abs(data['close'].shift(1) - data['close'].shift(2)).replace(0, np.nan))
    data['gap_medium'] = (data['close'] - data['close'].shift(5)) / (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)
    
    # Volume-weighted momentum
    vol_weighted_momentum = []
    for i in range(len(data)):
        if i >= 4:
            window_close = data['close'].iloc[i-4:i+1]
            window_volume = data['volume'].iloc[i-4:i+1]
            momentum_sum = ((window_close - window_close.shift(1)).iloc[1:] * window_volume.iloc[1:]).sum()
            volume_sum = window_volume.iloc[1:].sum()
            vol_weighted_momentum.append(momentum_sum / volume_sum if volume_sum != 0 else 0)
        else:
            vol_weighted_momentum.append(0)
    data['vol_weighted_mom'] = vol_weighted_momentum
    
    # Gap-Range Alignment
    data['gap_range_eff'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Gap persistence ratio
    gap_persistence = []
    for i in range(len(data)):
        if i >= 4:
            count = 0
            for j in range(i-4, i+1):
                if j >= 2:
                    gap_size = abs(data['open'].iloc[j] - data['close'].iloc[j-1])
                    daily_range = data['high'].iloc[j] - data['low'].iloc[j]
                    if daily_range > 0 and gap_size > 0.5 * daily_range:
                        count += 1
            gap_persistence.append(count / 5)
        else:
            gap_persistence.append(0)
    data['gap_persistence'] = gap_persistence
    
    # Gap Reversal Dynamics
    data['intraday_reversal'] = np.sign(data['open'] - data['close'].shift(1)) * abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Accelerated momentum decay
    data['momentum_decay'] = ((data['close'] - data['close'].shift(2)) ** 2 / abs(data['close'].shift(1) - data['close'].shift(3)).replace(0, np.nan)) - abs(data['close'] - data['close'].shift(1))
    
    # Price Fractal Dimension
    price_fractal = []
    for i in range(len(data)):
        if i >= 4:
            window_high = data['high'].iloc[i-4:i+1]
            window_low = data['low'].iloc[i-4:i+1]
            daily_range = data['high'].iloc[i] - data['low'].iloc[i]
            total_range = window_high.max() - window_low.min()
            if daily_range > 0 and total_range > 0:
                price_fractal.append(np.log(daily_range) / np.log(total_range))
            else:
                price_fractal.append(0)
        else:
            price_fractal.append(0)
    data['price_fractal'] = price_fractal
    
    # Volume Scaling
    volume_scaling = []
    for i in range(len(data)):
        if i >= 4:
            window_volume = data['volume'].iloc[i-4:i+1]
            median_vol = window_volume.median()
            if median_vol > 0 and data['volume'].iloc[i] > 0:
                volume_scaling.append(np.log(data['volume'].iloc[i]) / np.log(median_vol))
            else:
                volume_scaling.append(0)
        else:
            volume_scaling.append(0)
    data['volume_scaling'] = volume_scaling
    
    # Volume Efficiency
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Bid-ask pressure proxy
    data['bid_ask_pressure'] = ((2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)) * data['volume']
    
    # Order Flow Entropy
    data['order_flow_entropy'] = data['amount'] / (data['volume'] * data['close']).replace(0, np.nan)
    
    # Nested volatility ratio
    nested_vol = []
    for i in range(len(data)):
        if i >= 4:
            window_high = data['high'].iloc[i-4:i+1]
            window_low = data['low'].iloc[i-4:i+1]
            daily_range = data['high'].iloc[i] - data['low'].iloc[i]
            total_range = window_high.max() - window_low.min()
            if total_range > 0:
                nested_vol.append(daily_range / total_range)
            else:
                nested_vol.append(0)
        else:
            nested_vol.append(0)
    data['nested_vol_ratio'] = nested_vol
    
    # Volume Breakout
    volume_breakout = []
    for i in range(len(data)):
        if i >= 19:
            window_volume = data['volume'].iloc[i-19:i+1]
            avg_volume = window_volume.mean()
            if avg_volume > 0:
                volume_breakout.append(data['volume'].iloc[i] / avg_volume)
            else:
                volume_breakout.append(0)
        else:
            volume_breakout.append(0)
    data['volume_breakout'] = volume_breakout
    
    # Construct composite factors
    # Fractal Gap Momentum
    data['fractal_gap_momentum'] = (
        (data['gap_short'].fillna(0) + data['gap_medium'].fillna(0) + data['vol_weighted_mom'].fillna(0)) / 3 * 
        data['price_fractal'].fillna(0) * 
        data['volume_efficiency'].fillna(0) * 
        data['gap_persistence'].fillna(0)
    )
    
    # Efficiency-Weighted Reversal
    data['efficiency_reversal'] = (
        (data['intraday_reversal'].fillna(0) + data['momentum_decay'].fillna(0)) / 2 * 
        data['order_flow_entropy'].fillna(0) * 
        data['bid_ask_pressure'].fillna(0)
    )
    
    # Volatility-Association Alpha
    data['volatility_alpha'] = (
        data['gap_range_eff'].fillna(0) * 
        data['nested_vol_ratio'].fillna(0) * 
        data['volume_efficiency'].fillna(0)
    )
    
    # Core Fractal Gap Momentum
    data['core_fractal_gap'] = (
        data['fractal_gap_momentum'].fillna(0) * 
        data['gap_range_eff'].fillna(0) * 
        data['volume_breakout'].fillna(0)
    )
    
    # Association-Enhanced Reversal
    data['association_reversal'] = (
        data['efficiency_reversal'].fillna(0) * 
        data['bid_ask_pressure'].fillna(0) * 
        data['volume_scaling'].fillna(0)
    )
    
    # Final Alpha
    alpha = (
        data['core_fractal_gap'].fillna(0) * 
        data['association_reversal'].fillna(0) * 
        data['nested_vol_ratio'].fillna(0)
    )
    
    return alpha
