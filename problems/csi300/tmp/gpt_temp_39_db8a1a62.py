import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decay Quality Assessment
    # Multi-timeframe Momentum Gradient
    data['mom_2d'] = data['close'] / data['close'].shift(2) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Decay Acceleration Patterns
    data['decay_short'] = data['mom_2d'] - data['mom_5d']
    data['decay_medium'] = data['mom_5d'] - data['mom_10d']
    data['decay_intensity'] = data['decay_short'] * data['decay_medium']
    
    # Intraday Absorption Efficiency
    # Price Efficiency Components
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_persistence'] = data['open'] / data['close'].shift(1) - 1
    data['momentum_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Absorption Strength Metrics
    data['gap_absorption_strength'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['range_breakout_authenticity'] = (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low']).replace(0, np.nan)
    data['absorption_quality'] = data['gap_absorption_strength'] * data['range_breakout_authenticity']
    
    # Volume-Price Synchronization Analysis
    # Volume-Pressure Components
    intraday_pressure = []
    for i in range(len(data)):
        if i >= 10:
            window = data.iloc[i-9:i+1]
            pressure = sum((np.maximum(0, window['close'] - window['open']) - 
                          np.maximum(0, window['open'] - window['close'])) * window['volume'])
            intraday_pressure.append(pressure)
        else:
            intraday_pressure.append(np.nan)
    data['intraday_pressure'] = intraday_pressure
    
    data['volume_surprise'] = data['volume'] / data['volume'].shift(1)
    data['amount_momentum'] = data['amount'] / data['amount'].shift(1)
    
    # Synchronization Quality
    data['volume_amount_alignment'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['amount'] - data['amount'].shift(1))
    
    turnover_divergence = []
    for i in range(len(data)):
        if i >= 5:
            high_vol_ratio = (data['high'].iloc[i] * data['volume'].iloc[i]) / (data['high'].iloc[i-5] * data['volume'].iloc[i-5])
            low_vol_ratio = (data['low'].iloc[i] * data['volume'].iloc[i]) / (data['low'].iloc[i-5] * data['volume'].iloc[i-5])
            turnover_divergence.append(high_vol_ratio - low_vol_ratio)
        else:
            turnover_divergence.append(np.nan)
    data['turnover_divergence'] = turnover_divergence
    
    data['synchronization_strength'] = data['volume_surprise'] * data['amount_momentum'] * data['volume_amount_alignment']
    
    # Volatility-Adjusted Context
    # Volatility Breakout Detection
    data['range_expansion'] = ((data['high'] - data['low']) / data['close']) > (2.0 * (data['high'].shift(3) - data['low'].shift(3)) / data['close'].shift(3))
    data['volume_breakout'] = data['volume'] / data['volume'].shift(1) > 2.0
    data['pressure_alignment'] = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volatility-Adjusted Alignment
    price_volume_corr = []
    for i in range(len(data)):
        if i >= 5:
            window = data.iloc[i-4:i+1]
            price_returns = window['close'] / window['close'].shift(1) - 1
            volume_returns = window['volume'] / window['volume'].shift(1) - 1
            vol_std = window['high'].subtract(window['low']).std()
            if vol_std > 0:
                correlation = (price_returns * volume_returns).sum() / vol_std
            else:
                correlation = np.nan
            price_volume_corr.append(correlation)
        else:
            price_volume_corr.append(np.nan)
    data['price_volume_correlation'] = price_volume_corr
    
    data['context_multiplier'] = data['range_expansion'].astype(float) * data['volume_breakout'].astype(float)
    
    # Adaptive Signal Construction
    # Core Absorption Signal
    data['momentum_decay_absorption'] = data['decay_intensity'] * data['momentum_efficiency']
    data['volume_price_confirmation'] = data['synchronization_strength'] * data['pressure_alignment']
    data['absorption_quality_multiplier'] = data['absorption_quality'] * data['context_multiplier']
    
    # Final Factor Construction
    data['factor'] = (data['momentum_decay_absorption'] + 
                     data['volume_price_confirmation'] + 
                     data['absorption_quality_multiplier'])
    
    # Clean infinite values and handle missing data
    data['factor'] = data['factor'].replace([np.inf, -np.inf], np.nan)
    
    return data['factor']
