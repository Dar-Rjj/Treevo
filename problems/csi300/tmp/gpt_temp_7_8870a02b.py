import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Intraday Price Reversal Framework
    # Morning Reversal Signal
    data['Opening_Gap_Reversal'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + epsilon)
    data['Morning_Pressure'] = (data['high'] - data['open']) / (data['open'] - data['low'] + epsilon)
    data['Morning_Reversal_Core'] = data['Opening_Gap_Reversal'] * data['Morning_Pressure']
    
    # Afternoon Reversal Signal
    data['Midday_Momentum'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low']) + epsilon)
    data['Closing_Pressure'] = (data['close'] - data['low']) / (data['high'] - data['close'] + epsilon)
    data['Afternoon_Reversal_Core'] = data['Midday_Momentum'] * data['Closing_Pressure']
    
    # Intraday Reversal Divergence
    data['Reversal_Direction'] = np.sign(data['Morning_Reversal_Core']) * np.sign(data['Afternoon_Reversal_Core'])
    data['Intraday_Reversal_Strength'] = data['Morning_Reversal_Core'] - data['Afternoon_Reversal_Core']
    
    # Volume-Price Divergence Detection
    # Volume Acceleration Patterns
    data['Volume_Momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['Price_Volume_Divergence'] = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + epsilon) - data['Volume_Momentum']
    data['Acceleration_Signal'] = data['Volume_Momentum'] * data['Price_Volume_Divergence']
    
    # Intraday Volume Distribution
    data['Early_Volume_Concentration'] = data['volume'] * (data['open'] - data['low']) / (data['high'] - data['low'] + epsilon)
    data['Late_Volume_Concentration'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + epsilon)
    data['Volume_Timing_Divergence'] = data['Early_Volume_Concentration'] - data['Late_Volume_Concentration']
    
    # Volume-Price Timing Alignment
    data['Morning_Alignment'] = np.sign(data['Morning_Reversal_Core']) * np.sign(data['Early_Volume_Concentration'])
    data['Afternoon_Alignment'] = np.sign(data['Afternoon_Reversal_Core']) * np.sign(data['Late_Volume_Concentration'])
    data['Volume_Price_Timing_Score'] = data['Morning_Alignment'] * data['Afternoon_Alignment']
    
    # Multi-Period Reversal Persistence
    # Short-term Reversal Memory
    data['Previous_Day_Reversal'] = (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + epsilon)
    data['Two_Day_Reversal_Pattern'] = data['Previous_Day_Reversal'] * data['Morning_Reversal_Core']
    
    # Calculate rolling same-sign counts
    data['Reversal_Continuity'] = 0
    for i in range(3, len(data)):
        window = data['Two_Day_Reversal_Pattern'].iloc[i-3:i+1]
        if len(window) == 4:
            signs = np.sign(window)
            data.loc[data.index[i], 'Reversal_Continuity'] = np.sum(signs == signs.iloc[-1])
    
    # Volume Pattern Memory
    data['Volume_Direction_Persistence'] = 0
    for i in range(3, len(data)):
        window = data['Volume_Momentum'].iloc[i-3:i+1]
        if len(window) == 4:
            signs = np.sign(window)
            data.loc[data.index[i], 'Volume_Direction_Persistence'] = np.sum(signs == signs.iloc[-1])
    
    data['Volume_Magnitude_Trend'] = data['volume'] / (data['volume'].shift(3) + epsilon) - 1
    data['Volume_Memory_Score'] = data['Volume_Direction_Persistence'] * data['Volume_Magnitude_Trend']
    
    # Multi-Period Reversal Core
    data['Reversal_Volume_Integration'] = data['Reversal_Continuity'] * data['Volume_Memory_Score']
    data['Persistent_Reversal_Signal'] = data['Intraday_Reversal_Strength'] * data['Reversal_Volume_Integration']
    
    # Price Range Exhaustion Indicators
    # Range Utilization Metrics
    data['Upper_Range_Utilization'] = (data['high'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['Lower_Range_Utilization'] = (data['open'] - data['low']) / (data['high'] - data['low'] + epsilon)
    data['Range_Asymmetry_Score'] = data['Upper_Range_Utilization'] - data['Lower_Range_Utilization']
    
    # Volume-Range Congruence
    data['High_Volume_Range'] = data['volume'] * (data['high'] - data['close'].shift(1)) / (data['high'] - data['low'] + epsilon)
    data['Low_Volume_Range'] = data['volume'] * (data['close'].shift(1) - data['low']) / (data['high'] - data['low'] + epsilon)
    data['Volume_Range_Alignment'] = data['High_Volume_Range'] - data['Low_Volume_Range']
    
    # Exhaustion Detection Framework
    data['Range_Exhaustion_Signal'] = data['Range_Asymmetry_Score'] * data['Volume_Range_Alignment']
    data['Exhaustion_Confirmation'] = np.sign(data['Range_Exhaustion_Signal']) * np.sign(data['Intraday_Reversal_Strength'])
    
    # Signal Quality Assessment
    # Reversal Signal Quality
    data['Reversal_Magnitude_Consistency'] = data['Intraday_Reversal_Strength'].abs().rolling(window=3, min_periods=1).mean()
    data['Reversal_Direction_Stability'] = 0
    for i in range(3, len(data)):
        window = data['Intraday_Reversal_Strength'].iloc[i-3:i+1]
        if len(window) == 4:
            signs = np.sign(window)
            data.loc[data.index[i], 'Reversal_Direction_Stability'] = np.sum(signs == signs.iloc[-1])
    
    data['Reversal_Quality_Score'] = data['Reversal_Magnitude_Consistency'] * data['Reversal_Direction_Stability']
    
    # Volume Signal Quality
    data['Volume_Pattern_Consistency'] = 0
    for i in range(3, len(data)):
        window = data['Volume_Timing_Divergence'].iloc[i-3:i+1]
        if len(window) == 4:
            signs = np.sign(window)
            data.loc[data.index[i], 'Volume_Pattern_Consistency'] = np.sum(signs == signs.iloc[-1])
    
    data['Volume_Magnitude_Stability'] = (data['volume'] / data['volume'].shift(1)).rolling(window=3, min_periods=1).mean()
    data['Volume_Quality_Score'] = data['Volume_Pattern_Consistency'] * data['Volume_Magnitude_Stability']
    
    # Composite Quality Framework
    data['Signal_Reliability'] = data['Reversal_Quality_Score'] * data['Volume_Quality_Score']
    data['Quality_Enhanced_Reversal'] = data['Intraday_Reversal_Strength'] * data['Signal_Reliability']
    data['Quality_Enhanced_Volume'] = data['Volume_Price_Timing_Score'] * data['Signal_Reliability']
    
    # Final Alpha Construction
    # Core Component Integration
    data['Reversal_Volume_Core'] = data['Quality_Enhanced_Reversal'] * data['Quality_Enhanced_Volume']
    data['Persistence_Exhaustion_Core'] = data['Persistent_Reversal_Signal'] * data['Exhaustion_Confirmation']
    data['Timing_Alignment_Core'] = data['Volume_Price_Timing_Score'] * data['Reversal_Continuity']
    
    # Multi-Dimensional Synthesis
    data['Primary_Alpha_Component'] = data['Reversal_Volume_Core'] * data['Persistence_Exhaustion_Core']
    data['Secondary_Enhancement'] = data['Primary_Alpha_Component'] * data['Timing_Alignment_Core']
    data['Quality_Adjustment'] = data['Secondary_Enhancement'] * data['Signal_Reliability']
    
    # Final Alpha Output
    data['Alpha_Core'] = data['Quality_Adjustment'] * data['Volume_Memory_Score']
    data['Intraday_Momentum_Reversal_Alpha'] = data['Alpha_Core'] * data['Range_Exhaustion_Signal']
    
    # Return the final alpha factor
    return data['Intraday_Momentum_Reversal_Alpha']
