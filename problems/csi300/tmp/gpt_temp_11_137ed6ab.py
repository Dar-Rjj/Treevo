import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Volume Cointegration with Fractal Breakout Confirmation
    """
    data = df.copy()
    
    # Multi-Scale Volatility Framework
    data['True_Range_Volatility'] = (data['high'] - data['low']) / data['close']
    data['Gap_Volatility'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['Volatility_Ratio'] = data['True_Range_Volatility'] / data['Gap_Volatility'].replace(0, np.nan)
    
    # Multi-Scale Volume Dynamics
    data['Volume_Momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['Volume_Clustering'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    data['Volume_Fractal'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Gap-Momentum Cointegration
    data['Overnight_Gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['Intraday_Gap'] = (data['close'] - data['open']) / data['open']
    data['Gap_Asymmetry'] = data['Overnight_Gap'] - data['Intraday_Gap']
    data['Gap_Momentum_Divergence'] = ((data['close'] - data['close'].shift(2)) / data['close'].shift(2)) - ((data['close'] - data['close'].shift(6)) / data['close'].shift(6))
    
    # Fractal Breakout System
    data['Current_Session_Range'] = data['high'] - data['low']
    data['Previous_Session_Range'] = data['high'].shift(1) - data['low'].shift(1)
    data['Range_Expansion'] = data['Current_Session_Range'] / data['Previous_Session_Range']
    
    # Calculate rolling highs and lows for breakout components
    data['Rolling_High_4'] = data['high'].rolling(window=4, min_periods=4).apply(lambda x: x[:-1].max() if len(x) == 4 else np.nan)
    data['Rolling_Low_4'] = data['low'].rolling(window=4, min_periods=4).apply(lambda x: x[:-1].min() if len(x) == 4 else np.nan)
    
    data['Upper_Breakout'] = (data['high'] - data['Rolling_High_4']) / data['close'].shift(1)
    data['Lower_Breakout'] = (data['Rolling_Low_4'] - data['low']) / data['close'].shift(1)
    data['Breakout_Direction'] = np.sign(data['close'] - data['open']) * np.sign(data['high'] - data['high'].shift(1))
    
    # Pressure-Cointegration Dynamics
    data['Upper_Pressure'] = (data['high'] - data['close']) / data['Current_Session_Range'].replace(0, np.nan)
    data['Lower_Pressure'] = (data['close'] - data['low']) / data['Current_Session_Range'].replace(0, np.nan)
    data['Pressure_Imbalance'] = data['Upper_Pressure'] - data['Lower_Pressure']
    
    data['Gap_Pressure'] = data['Overnight_Gap'] * data['Pressure_Imbalance']
    data['Volume_Pressure'] = data['Volume_Momentum'] * data['Pressure_Imbalance']
    data['Volatility_Pressure'] = data['Volatility_Ratio'] * data['Pressure_Imbalance']
    
    # Fractal Regime Detection
    data['Volatility_Regime'] = data['True_Range_Volatility'] > data['True_Range_Volatility'].rolling(window=4, min_periods=4).mean()
    
    def volume_regime(x):
        if x > 1.2:
            return 'Clustered'
        elif x < 0.8:
            return 'Dispersed'
        else:
            return 'Normal'
    
    data['Volume_Regime'] = data['Volume_Clustering'].apply(volume_regime)
    
    def cointegration_regime(x):
        if x > 0:
            return 'Convergent'
        elif x < 0:
            return 'Divergent'
        else:
            return 'Neutral'
    
    data['Cointegration_Regime'] = (data['Gap_Momentum_Divergence'] * data['Volume_Fractal']).apply(cointegration_regime)
    
    # Adaptive Factor Construction
    # High Volatility Regime Factors
    data['High_Vol_Clustered'] = data['Volatility_Ratio'] * data['Volume_Clustering'] * data['Gap_Pressure']
    data['High_Vol_Dispersed'] = data['Volume_Fractal'] * data['Breakout_Direction'] * data['Volatility_Pressure']
    data['High_Vol_Normal'] = data['Range_Expansion'] * data['Volume_Pressure'] * data['Gap_Asymmetry']
    
    # Low Volatility Regime Factors
    data['Low_Vol_Clustered'] = data['Gap_Momentum_Divergence'] * data['Volume_Clustering'] * data['Intraday_Gap']
    data['Low_Vol_Dispersed'] = data['Volume_Fractal'] * data['Pressure_Imbalance'] * data['Overnight_Gap']
    data['Low_Vol_Normal'] = data['Breakout_Direction'] * data['Volume_Momentum'] * data['Gap_Asymmetry']
    
    # Regime-Adaptive Selection
    def select_factor(row):
        if row['Volatility_Regime']:
            if row['Volume_Regime'] == 'Clustered':
                return row['High_Vol_Clustered']
            elif row['Volume_Regime'] == 'Dispersed':
                return row['High_Vol_Dispersed']
            else:
                return row['High_Vol_Normal']
        else:
            if row['Volume_Regime'] == 'Clustered':
                return row['Low_Vol_Clustered']
            elif row['Volume_Regime'] == 'Dispersed':
                return row['Low_Vol_Dispersed']
            else:
                return row['Low_Vol_Normal']
    
    data['Selected_Factor'] = data.apply(select_factor, axis=1)
    
    # Cointegration Regime Adjustment
    def cointegration_adjustment(row):
        if row['Cointegration_Regime'] == 'Convergent':
            return row['Selected_Factor'] * 1.2
        elif row['Cointegration_Regime'] == 'Divergent':
            return row['Selected_Factor'] * 0.8
        else:
            return row['Selected_Factor']
    
    data['Adjusted_Factor'] = data.apply(cointegration_adjustment, axis=1)
    
    # Final Factor Enhancement
    data['Final_Factor'] = data['Adjusted_Factor'] * (1 + data['Breakout_Direction'] * data['Range_Expansion'])
    
    return data['Final_Factor']
