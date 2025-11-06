import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum and Reversal Detection
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['reversal_pattern'] = (data['short_term_momentum'] < 0) & (data['medium_term_momentum'] > 0)
    
    # Volume Divergence Analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_divergence'] = data['volume_momentum'] * data['short_term_momentum']
    
    # Intraday Pressure Dynamics
    data['net_pressure'] = (data['close'] - data['low']) - (data['high'] - data['close'])
    
    # Pressure Cumulation using rolling window
    def pressure_cumulation_func(x):
        if len(x) < 3:
            return np.nan
        net_pressure = x['net_pressure'].values
        volume = x['volume'].values
        return np.sum(net_pressure * volume) / np.sum(volume) if np.sum(volume) > 0 else 0
    
    # Create rolling window for pressure cumulation
    pressure_cumulation_values = []
    for i in range(len(data)):
        if i < 2:
            pressure_cumulation_values.append(np.nan)
        else:
            window_data = data.iloc[i-2:i+1][['net_pressure', 'volume']]
            pressure_cumulation_values.append(pressure_cumulation_func(window_data))
    
    data['pressure_cumulation'] = pressure_cumulation_values
    
    # Composite Factor Generation
    data['base_reversal_signal'] = data['short_term_momentum'] * data['medium_term_momentum']
    data['volume_confirmation'] = data['volume_divergence'] * data['volume_momentum']
    
    # Final Alpha calculation
    data['alpha_factor'] = (data['base_reversal_signal'] * 
                           data['volume_confirmation'] * 
                           (data['net_pressure'] * data['pressure_cumulation']))
    
    # Return the alpha factor series
    return data['alpha_factor']
