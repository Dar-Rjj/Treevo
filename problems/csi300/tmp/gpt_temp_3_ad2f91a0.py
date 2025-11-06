import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Pressure Accumulation Factor
    Combines price pressure, volume confirmation, and overnight gap persistence
    to generate a predictive trading signal.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['high_low_range'] = data['high'] - data['low']
    data['high_low_range'] = np.where(data['high_low_range'] == 0, 1e-6, data['high_low_range'])  # Avoid division by zero
    
    # Calculate upside pressure component
    data['upside_price_pressure'] = (data['high'] - data['open']) / data['high_low_range']
    
    # Calculate downside pressure component  
    data['downside_price_pressure'] = (data['open'] - data['low']) / data['high_low_range']
    
    # Calculate volume ratios
    data['upside_volume_ratio'] = np.where(data['close'] > data['prev_close'], data['volume'], 0)
    total_volume = data['volume'].rolling(window=5, min_periods=1).sum()
    data['upside_volume_ratio'] = data['upside_volume_ratio'].rolling(window=5, min_periods=1).sum() / total_volume
    
    data['downside_volume_ratio'] = np.where(data['close'] < data['prev_close'], data['volume'], 0)
    data['downside_volume_ratio'] = data['downside_volume_ratio'].rolling(window=5, min_periods=1).sum() / total_volume
    
    # Apply volume ratios to pressure components
    data['upside_pressure'] = data['upside_price_pressure'] * data['upside_volume_ratio']
    data['downside_pressure'] = data['downside_price_pressure'] * data['downside_volume_ratio']
    
    # Calculate net pressure
    data['net_pressure'] = data['upside_pressure'] - data['downside_pressure']
    
    # Cumulative sum with zero-crossing reset over 3-day window
    pressure_accumulation = []
    current_accumulation = 0
    
    for i in range(len(data)):
        if i == 0:
            current_accumulation = data['net_pressure'].iloc[i]
        else:
            # Check for zero crossing in the last 3 days
            recent_pressures = data['net_pressure'].iloc[max(0, i-2):i+1]
            zero_crossings = ((recent_pressures.shift(1) * recent_pressures) < 0).sum()
            
            if zero_crossings > 0:
                current_accumulation = data['net_pressure'].iloc[i]
            else:
                current_accumulation += data['net_pressure'].iloc[i]
        
        pressure_accumulation.append(current_accumulation)
    
    data['pressure_accumulation'] = pressure_accumulation
    
    # Volume confirmation strength
    data['pressure_days'] = (data['net_pressure'].abs() > data['net_pressure'].rolling(window=10).std()).astype(int)
    data['pressure_day_ratio'] = data['pressure_days'].rolling(window=10, min_periods=5).mean()
    data['volume_confirmation'] = np.sqrt(data['pressure_day_ratio'])
    
    # Overnight gap persistence
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_persistence'] = np.sign(data['overnight_gap']) * np.sign(data['pressure_accumulation'])
    data['gap_persistence'] = np.where(data['gap_persistence'] > 0, 1 + data['overnight_gap'].abs(), 
                                     1 - data['overnight_gap'].abs())
    
    # Final factor calculation
    data['factor'] = (data['pressure_accumulation'] * data['volume_confirmation'] * data['gap_persistence'])
    
    # Normalize the factor
    data['factor'] = (data['factor'] - data['factor'].rolling(window=20).mean()) / data['factor'].rolling(window=20).std()
    
    return data['factor']
