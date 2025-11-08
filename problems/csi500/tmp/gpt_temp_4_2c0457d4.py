import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Pressure Accumulation Factor
    Combines price pressure components with volume validation to detect accumulation patterns
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price pressure components
    # Opening gap pressure: (open - previous close) / previous close
    data['prev_close'] = data['close'].shift(1)
    data['gap_pressure'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Intraday absorption: (close - open) / (high - low)
    data['intraday_absorption'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_absorption'] = data['intraday_absorption'].replace([np.inf, -np.inf], 0)
    
    # Closing pressure: (close - (high + low)/2) / ((high - low)/2)
    data['closing_pressure'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2)
    data['closing_pressure'] = data['closing_pressure'].replace([np.inf, -np.inf], 0)
    
    # Volume validation
    # Relative volume intensity: volume / 20-day median volume
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=1).median()
    data['volume_intensity'] = data['volume'] / data['volume_median_20d']
    
    # Volume distribution: (volume at close - volume at open) / total volume
    # Since we don't have intraday volume data, we'll approximate using amount
    data['volume_distribution'] = (data['amount'] - data['open'] * data['volume']) / data['amount']
    data['volume_distribution'] = data['volume_distribution'].replace([np.inf, -np.inf], 0)
    
    # Volume persistence: 5-day volume trend consistency
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0
    )
    
    # Pressure accumulation
    # Daily pressure score: weighted combination of price pressure components
    weights = [0.3, 0.4, 0.3]  # gap, absorption, closing
    data['daily_pressure'] = (
        weights[0] * data['gap_pressure'] + 
        weights[1] * data['intraday_absorption'] + 
        weights[2] * data['closing_pressure']
    )
    
    # Volume confirmation multiplier: product of volume validation metrics
    data['volume_multiplier'] = (
        data['volume_intensity'] * 
        (1 + data['volume_distribution']) * 
        (1 + data['volume_trend'])
    )
    
    # Accumulation period: 3-day rolling sum of confirmed pressure
    data['confirmed_pressure'] = data['daily_pressure'] * data['volume_multiplier']
    data['accumulated_pressure'] = data['confirmed_pressure'].rolling(window=3, min_periods=1).sum()
    
    # Final factor
    # Multiply accumulated pressure by current day's absorption
    # Apply volume persistence as confidence weight
    factor = data['accumulated_pressure'] * data['intraday_absorption'] * (1 + data['volume_trend'])
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
