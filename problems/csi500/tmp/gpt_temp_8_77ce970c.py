import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Momentum Acceleration
    # Price Momentum Structure
    data['Momentum_Velocity'] = (data['close'] - data['close'].shift(5)) / 5
    data['Momentum_Acceleration'] = (data['close'] - 2 * data['close'].shift(5) + data['close'].shift(10)) / 25
    data['Acceleration_Quality'] = np.sign(data['Momentum_Velocity']) * data['Momentum_Acceleration']
    data['Acceleration_Score'] = data['Momentum_Velocity'] * data['Momentum_Acceleration']
    
    # Volume Momentum Confirmation
    data['Volume_Momentum'] = data['volume'] / data['volume'].shift(5)
    data['Volume_Acceleration'] = (data['volume'] - 2 * data['volume'].shift(5) + data['volume'].shift(10)) / data['volume'].shift(10).replace(0, np.nan)
    data['Volume_Price_Alignment'] = np.sign(data['close'] - data['close'].shift(5)) * np.sign(data['volume'] - data['volume'].shift(5))
    data['Confirmation_Strength'] = data['Volume_Price_Alignment'] * data['Volume_Acceleration']
    
    # Intraday Momentum Quality
    data['Opening_Momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan)
    data['Intraday_Momentum'] = (data['close'] - data['open']) / data['open'].replace(0, np.nan)
    data['Momentum_Persistence'] = np.sign(data['Opening_Momentum']) * np.sign(data['Intraday_Momentum'])
    data['Day_Structure_Score'] = data['Momentum_Persistence'] * data['Intraday_Momentum']
    
    # Range Efficiency Dynamics
    # Volatility-Range Quality
    data['True_Range_Efficiency'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['Volatility_Normalized_Move'] = (data['close'] - data['open']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['Efficiency_Ratio'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['Range_Quality'] = data['True_Range_Efficiency'] * data['Efficiency_Ratio']
    
    # Asymmetric Range Microstructure
    data['Upside_Rejection_Quality'] = ((data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low']).replace(0, np.nan) * 
                                       (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan))
    data['Downside_Support_Quality'] = ((np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low']).replace(0, np.nan) * 
                                       (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan))
    data['Gap_Efficiency_Momentum'] = ((data['close'] - np.minimum(data['open'], data['close'].shift(1))) / 
                                     (np.maximum(data['open'], data['close'].shift(1)) - np.minimum(data['open'], data['close'].shift(1))).replace(0, np.nan) * 
                                     (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan))
    
    # Range-Enhanced Price Discovery
    data['Opening_Efficiency'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan) * 
                                (data['high'] - data['low']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan))
    data['Closing_Efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan) * 
                                (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2)).replace(0, np.nan))
    data['Intraday_Range_Quality'] = (abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan) * 
                                    (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan))
    
    # Volume-Range Convergence
    # Volume-Range Alignment
    data['Volume_Range_Concentration'] = data['volume'] * abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['Volume_Momentum_Quality'] = ((data['volume'] / data['volume'].shift(5) - 1) - 
                                     (data['volume'] / data['volume'].shift(10) - 1) * np.sign(data['volume'] - data['volume'].shift(1)))
    
    # Calculate Volume-Range Persistence
    vol_range_sign = np.sign(data['Volume_Momentum_Quality'] * data['Volume_Range_Concentration'])
    vol_range_persistence = vol_range_sign.groupby(vol_range_sign.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False
    ).reset_index(level=0, drop=True)
    data['Volume_Range_Persistence'] = vol_range_persistence
    
    # Range-Volume Confirmation
    data['Opening_Range_Confirmation'] = (data['volume'] / data['volume'].shift(1) * 
                                        (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan) * 
                                        data['Opening_Efficiency'])
    data['Closing_Range_Confirmation'] = (data['volume'] / data['volume'].shift(1) * 
                                        (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan) * 
                                        data['Closing_Efficiency'])
    data['Intraday_Range_Volume_Quality'] = (data['Volume_Range_Concentration'] * data['Intraday_Range_Quality'] * 
                                           data['Volume_Range_Persistence'])
    
    # Volume Microstructure Enhancement
    data['Price_Volume_Efficiency'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    vol_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    data['Volume_Spike_Detection'] = (data['close'] - data['open']) * (data['volume'] / vol_avg_5)
    data['Volume_Price_Alignment_Final'] = data['Price_Volume_Efficiency'] * data['Volume_Spike_Detection'] * (data['volume'] / data['volume'].shift(1))
    
    # Momentum-Range Integration
    # Range-Weighted Momentum
    data['Volume_Range_Weighted_Momentum'] = ((data['close'] - data['close'].shift(1)) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan) * 
                                            abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan))
    data['Efficient_Range_Discovery_Momentum'] = ((data['close'] - data['open']) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan) * 
                                                (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan))
    data['Gap_Range_Momentum_Quality'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan) * 
                                        data['Gap_Efficiency_Momentum'] * data['volume'] / data['volume'].shift(1))
    
    # Multi-Timeframe Consistency
    data['Short_term_Range_Momentum'] = ((data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2)).replace(0, np.nan) * 
                                       data['Volume_Range_Weighted_Momentum'])
    data['Medium_term_Range_Momentum'] = ((data['close'] - data['close'].shift(5)) / (data['close'].shift(5) - data['close'].shift(10)).replace(0, np.nan) * 
                                        data['Efficient_Range_Discovery_Momentum'])
    
    # Calculate Range-Momentum Persistence
    range_momentum_sign = np.sign(data['Short_term_Range_Momentum'] * data['Medium_term_Range_Momentum'])
    range_momentum_persistence = range_momentum_sign.groupby(range_momentum_sign.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False
    ).reset_index(level=0, drop=True)
    data['Range_Momentum_Persistence'] = range_momentum_persistence
    
    # Regime-Enhanced Signals
    data['High_Range_Momentum'] = data['Volume_Range_Weighted_Momentum'] * data['Upside_Rejection_Quality']
    data['Low_Range_Momentum'] = data['Efficient_Range_Discovery_Momentum'] * data['Downside_Support_Quality']
    data['Transition_Range_Momentum'] = data['Gap_Range_Momentum_Quality'] * data['Range_Momentum_Persistence']
    
    # Alpha Synthesis
    # Core Momentum Components
    data['Momentum_Acceleration_Core'] = data['Acceleration_Score'] * data['Confirmation_Strength']
    data['Range_Efficiency_Core'] = data['Day_Structure_Score'] * data['Range_Quality']
    data['Volume_Range_Core'] = data['Volume_Range_Concentration'] * data['Intraday_Range_Volume_Quality']
    
    # Convergence Enhancement
    data['Momentum_Range_Alignment'] = data['High_Range_Momentum'] * data['Low_Range_Momentum'] * data['Transition_Range_Momentum']
    data['Volume_Microstructure'] = data['Volume_Price_Alignment_Final'] * data['Volume_Spike_Detection'] * data['Price_Volume_Efficiency']
    data['Range_Quality_Enhancement'] = data['Upside_Rejection_Quality'] * data['Downside_Support_Quality'] * data['Gap_Efficiency_Momentum']
    
    # Integration Layer
    data['Momentum_Integration'] = data['Momentum_Acceleration_Core'] * data['Momentum_Range_Alignment']
    data['Volume_Integration'] = data['Volume_Microstructure'] * data['Volume_Range_Core']
    data['Range_Integration'] = data['Range_Quality_Enhancement'] * data['Range_Efficiency_Core']
    
    # Final Alpha
    data['Composite'] = data['Momentum_Integration'] * data['Volume_Integration'] * data['Range_Integration']
    alpha = data['Composite'] * np.sign(data['Range_Momentum_Persistence'])
    
    return alpha
