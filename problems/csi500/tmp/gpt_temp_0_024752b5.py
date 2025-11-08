import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volume Distribution Analysis
    # Intraday Volume Concentration
    data['Morning_Volume_Density'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['Afternoon_Volume_Density'] = data['volume'] / (data['close'] - data['open']).replace(0, np.nan)
    data['Volume_Concentration_Ratio'] = data['Morning_Volume_Density'] / data['Afternoon_Volume_Density'].replace(0, np.nan)
    
    # Volume Profile Asymmetry
    data['Upper_Volume_Pressure'] = (data['high'] - data['close']) * data['volume']
    data['Lower_Volume_Support'] = (data['close'] - data['low']) * data['volume']
    data['Volume_Pressure_Ratio'] = data['Upper_Volume_Pressure'] / data['Lower_Volume_Support'].replace(0, np.nan)
    
    # Volume Momentum Structure
    data['Volume_Acceleration'] = (data['volume'] / data['volume'].shift(1) - 
                                 data['volume'].shift(1) / data['volume'].shift(2))
    data['Volume_Persistence'] = data['volume'] / data['volume'].shift(3)
    data['Volume_Momentum_Score'] = data['Volume_Acceleration'] * data['Volume_Persistence']
    
    # Price Compression Dynamics
    # Range Compression Analysis
    data['Current_Range_Compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['Historical_Compression_Baseline'] = data['high'].rolling(window=10).apply(lambda x: (x - data.loc[x.index, 'low']).mean(), raw=False)
    data['Compression_Deviation'] = data['Current_Range_Compression'] / data['Historical_Compression_Baseline'].replace(0, np.nan)
    
    # Price Cluster Detection
    data['Opening_Cluster_Distance'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['Closing_Cluster_Distance'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['Cluster_Cohesion'] = data['Opening_Cluster_Distance'] * data['Closing_Cluster_Distance']
    
    # Compression Momentum
    data['Compression_Velocity'] = (data['Current_Range_Compression'] - 
                                  (data['high'].shift(2) - data['low'].shift(2)) / 
                                  (data['high'].shift(3) - data['low'].shift(3)).replace(0, np.nan))
    data['Compression_Acceleration'] = (data['Compression_Velocity'] - 
                                      ((data['high'].shift(3) - data['low'].shift(3)) / 
                                       (data['high'].shift(4) - data['low'].shift(4)).replace(0, np.nan) - 
                                       (data['high'].shift(4) - data['low'].shift(4)) / 
                                       (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)))
    data['Compression_Momentum_Score'] = data['Compression_Velocity'] * data['Compression_Acceleration']
    
    # Volume-Price Integration
    # Density-Compression Alignment
    data['Volume_Density_Momentum'] = data['Volume_Concentration_Ratio'] * data['Volume_Momentum_Score']
    data['Compression_Intensity'] = data['Compression_Deviation'] * data['Compression_Momentum_Score']
    data['Density_Compression_Signal'] = data['Volume_Density_Momentum'] * data['Compression_Intensity']
    
    # Pressure-Cluster Dynamics
    data['Volume_Pressure_Momentum'] = data['Volume_Pressure_Ratio'] * data['Volume_Momentum_Score']
    data['Cluster_Momentum'] = data['Cluster_Cohesion'] * data['Compression_Momentum_Score']
    data['Pressure_Cluster_Signal'] = data['Volume_Pressure_Momentum'] * data['Cluster_Momentum']
    
    # Integrated Volume-Price Momentum
    data['Primary_Momentum'] = data['Density_Compression_Signal'] * np.sign(data['Pressure_Cluster_Signal'])
    data['Secondary_Momentum'] = data['Pressure_Cluster_Signal'] * np.sign(data['Density_Compression_Signal'])
    data['Volume_Price_Factor'] = data['Primary_Momentum'] + data['Secondary_Momentum']
    
    # Regime-Based Signal Enhancement
    # Regime Classification
    data['High_Compression'] = data['Compression_Deviation'] < 0.7
    data['Low_Compression'] = data['Compression_Deviation'] > 1.3
    data['Transition_Regime'] = (data['Compression_Deviation'] >= 0.7) & (data['Compression_Deviation'] <= 1.3)
    
    # High Compression Regime
    data['Compression_Breakout_Potential'] = data['Volume_Price_Factor'] * abs(data['Compression_Deviation'])
    data['Density_Amplification'] = data['Compression_Breakout_Potential'] * data['Volume_Concentration_Ratio']
    data['High_Compression_Factor'] = data['Density_Amplification'] * data['Volume_Pressure_Ratio']
    
    # Low Compression Regime
    data['Range_Expansion_Confirmation'] = data['Volume_Price_Factor'] * data['Current_Range_Compression']
    data['Volume_Validation'] = data['Range_Expansion_Confirmation'] * data['Volume_Momentum_Score']
    data['Low_Compression_Factor'] = data['Volume_Validation'] * data['Cluster_Cohesion']
    
    # Transition Regime
    data['Momentum_Persistence'] = data['Volume_Price_Factor'] - data['Volume_Price_Factor'].shift(2)
    data['Compression_Stability'] = abs(data['Current_Range_Compression'] - 1)
    data['Transition_Factor'] = data['Momentum_Persistence'] / data['Compression_Stability'].replace(0, np.nan)
    
    # Dynamic Factor Construction
    # Adaptive Signal Processing
    data['Regime_Adapted_Base'] = np.where(
        data['High_Compression'], data['High_Compression_Factor'],
        np.where(data['Low_Compression'], data['Low_Compression_Factor'], data['Transition_Factor'])
    )
    data['Volume_Confirmation'] = data['Regime_Adapted_Base'] * data['Volume_Momentum_Score']
    data['Compression_Momentum_Enhancement'] = data['Volume_Confirmation'] * data['Compression_Momentum_Score']
    
    # Final Alpha Generation
    data['Core_Factor'] = data['Compression_Momentum_Enhancement']
    data['Persistence_Filter'] = data['Core_Factor'] * np.sign(data['Volume_Price_Factor'] - data['Volume_Price_Factor'].shift(1))
    data['Volume_Density_Alpha'] = data['Persistence_Filter'] * abs(data['Volume_Concentration_Ratio'])
    
    return data['Volume_Density_Alpha']
