import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Directional Momentum Components
    data['Bullish_Momentum_Intensity'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['Bearish_Momentum_Intensity'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['Momentum_Asymmetry'] = data['Bullish_Momentum_Intensity'] - data['Bearish_Momentum_Intensity']
    
    # Gap Momentum Score
    data['Gap_Momentum_Score'] = ((data['open'] / data['close'].shift(1) - 1) * 
                                 (data['close'] - data['open']) / (data['close'].shift(1) - data['open'].shift(1)).replace(0, np.nan))
    
    # Volume-Weighted Momentum
    data['Volume_Enhanced_Bullish'] = data['Bullish_Momentum_Intensity'] * (data['volume'] / data['volume'].shift(1)).replace([np.inf, -np.inf], np.nan)
    data['Volume_Enhanced_Bearish'] = data['Bearish_Momentum_Intensity'] * (data['volume'] / data['volume'].shift(1)).replace([np.inf, -np.inf], np.nan)
    data['Volume_Momentum_Divergence'] = data['Volume_Enhanced_Bullish'] - data['Volume_Enhanced_Bearish']
    
    # Liquidity Pressure
    data['Liquidity_Pressure'] = ((data['high'] + data['low'] - 2 * data['close'].shift(1)) / 
                                 (data['high'] - data['low']).replace(0, np.nan) * 
                                 (data['volume'] / data['volume'].shift(1)).replace([np.inf, -np.inf], np.nan))
    
    # Multi-Timeframe Momentum
    data['Short_Term_Momentum'] = ((data['close'] / data['close'].shift(2) - 1) / 
                                  (data['close'] / data['close'].shift(1) - 1).replace(0, np.nan))
    data['Medium_Term_Momentum'] = ((data['close'] / data['close'].shift(5) - 1) / 
                                   (data['close'] / data['close'].shift(3) - 1).replace(0, np.nan))
    data['Momentum_Acceleration'] = data['Short_Term_Momentum'] - data['Medium_Term_Momentum']
    data['Momentum_Persistence'] = ((data['close'] / data['close'].shift(3) - 1) / 
                                   (data['close'] / data['close'].shift(6) - 1).replace(0, np.nan) * 
                                   np.sign(data['close'] / data['close'].shift(1) - 1))
    
    # Fractal Range Analysis
    data['Fractal_Range_Ratio'] = ((data['high'] - data['low']) / 
                                  (data['high'].shift(2) - data['low'].shift(2)).replace(0, np.nan))
    data['Range_Expansion'] = (data['Fractal_Range_Ratio'] / 
                              ((data['high'].shift(1) - data['low'].shift(1)) / 
                               (data['high'].shift(3) - data['low'].shift(3)).replace(0, np.nan)).replace(0, np.nan))
    
    # Range Persistence
    data['Range_Persistence'] = (data['Fractal_Range_Ratio'].rolling(window=3, min_periods=1)
                                .apply(lambda x: (x > 1).sum(), raw=False))
    
    # Net Breakout Velocity
    data['Net_Breakout_Velocity'] = (((data['high'] - data['close'].shift(1)) - (data['close'].shift(1) - data['low'])) * 
                                    data['volume'] * 
                                    (data['close'] / data['close'].shift(5) - 1 - (data['close'] / data['close'].shift(10) - 1)))
    
    # Breakout Signal Generation
    data['Upper_Breakout_Potential'] = ((data['high'] - data['close'].shift(1)) / 
                                       (data['high'] - data['low']).replace(0, np.nan))
    data['Lower_Breakout_Potential'] = ((data['close'].shift(1) - data['low']) / 
                                       (data['high'] - data['low']).replace(0, np.nan))
    data['Breakout_Asymmetry'] = data['Upper_Breakout_Potential'] - data['Lower_Breakout_Potential']
    data['Breakout_Confirmation'] = data['Breakout_Asymmetry'] * data['Range_Expansion']
    
    # Fractal Pattern Recognition
    data['Fractal_Resistance'] = ((data['high'] - data['open']) / 
                                 (data['open'] - data['low']).replace(0, np.nan))
    data['Fractal_Support'] = ((data['close'] - data['low']) / 
                              (data['high'] - data['close']).replace(0, np.nan))
    data['Fractal_Strength'] = data['Fractal_Resistance'] - data['Fractal_Support']
    data['Intraday_Efficiency_Asymmetry'] = data['Fractal_Resistance'] - data['Fractal_Support']
    
    # Volume Distribution Analysis
    data['Up_Day_Volume_Concentration'] = np.where(data['close'] > data['close'].shift(1), data['volume'], 0) / data['volume'].replace(0, np.nan)
    data['Down_Day_Volume_Concentration'] = np.where(data['close'] < data['close'].shift(1), data['volume'], 0) / data['volume'].replace(0, np.nan)
    data['Volume_Concentration_Ratio'] = data['Up_Day_Volume_Concentration'] / data['Down_Day_Volume_Concentration'].replace(0, np.nan)
    
    # Volume Pressure Score
    data['Volume_Pressure_Score'] = (((data['volume'] / data['volume'].shift(3) - 1) / 
                                    (data['volume'] / data['volume'].shift(5) - 1).replace(0, np.nan)) * 
                                    np.sign(data['volume'] / data['volume'].shift(3) - 1))
    
    # Amount-Volume Dynamics
    data['Amount_Efficiency'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['Volume_Efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['Amount_Volume_Ratio'] = data['Amount_Efficiency'] / data['Volume_Efficiency'].replace(0, np.nan)
    
    # Price-Volume Divergence
    data['Price_Volume_Divergence'] = (((data['close'] / data['close'].shift(3) - 1) / 
                                      (data['volume'] / data['volume'].shift(3) - 1).replace(0, np.nan)) * 
                                      np.sign(data['close'] / data['close'].shift(1) - 1))
    
    # Asymmetric Volume Pressure
    data['Bullish_Volume_Pressure'] = data['Up_Day_Volume_Concentration'] * data['Volume_Efficiency']
    data['Bearish_Volume_Pressure'] = data['Down_Day_Volume_Concentration'] * data['Volume_Efficiency']
    data['Net_Volume_Pressure'] = data['Bullish_Volume_Pressure'] - data['Bearish_Volume_Pressure']
    
    # Volume Pressure Momentum
    data['Volume_Pressure_Momentum'] = (data['Net_Volume_Pressure'] / 
                                       (data['Bullish_Volume_Pressure'].shift(1) - data['Bearish_Volume_Pressure'].shift(1)).replace(0, np.nan))
    
    # Signal Integration & Enhancement
    data['Momentum_Breakout_Convergence'] = data['Momentum_Asymmetry'] * data['Breakout_Asymmetry']
    data['Volume_Enhanced_Breakout'] = data['Breakout_Confirmation'] * data['Volume_Momentum_Divergence']
    data['Fractal_Momentum_Integration'] = data['Fractal_Strength'] * data['Momentum_Persistence']
    data['Breakout_Momentum_Score'] = data['Momentum_Breakout_Convergence'] + data['Volume_Enhanced_Breakout']
    
    # Volume-Price Synchronization
    data['Volume_Price_Alignment'] = np.sign(data['Momentum_Asymmetry']) * np.sign(data['Net_Volume_Pressure'])
    data['Amount_Volume_Synchronization'] = data['Amount_Volume_Ratio'] / data['Amount_Volume_Ratio'].shift(1).replace(0, np.nan)
    data['Synchronization_Strength'] = data['Volume_Price_Alignment'] * data['Amount_Volume_Synchronization']
    data['Volume_Price_Momentum'] = data['Synchronization_Strength'] * data['Price_Volume_Divergence']
    
    # Multi-Factor Enhancement
    data['Core_Momentum_Factor'] = data['Gap_Momentum_Score'] * data['Momentum_Acceleration']
    data['Volume_Pressure_Factor'] = data['Net_Volume_Pressure'] * data['Volume_Pressure_Score']
    data['Breakout_Intensity_Factor'] = data['Net_Breakout_Velocity'] * data['Range_Expansion']
    data['Synchronization_Factor'] = data['Volume_Price_Momentum'] * data['Liquidity_Pressure']
    
    # Regime-Independent Components
    data['Momentum_Core'] = data['Core_Momentum_Factor'] * data['Momentum_Breakout_Convergence']
    data['Volume_Core'] = data['Volume_Pressure_Factor'] * data['Volume_Concentration_Ratio']
    data['Breakout_Core'] = data['Breakout_Intensity_Factor'] * data['Breakout_Confirmation']
    data['Sync_Core'] = data['Synchronization_Factor'] * data['Intraday_Efficiency_Asymmetry']
    
    # Dynamic Weighting Framework
    def count_sign_consistency(series, window):
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: (np.sign(x) == np.sign(x.shift(1))).sum() if len(x) > 1 else 1, raw=False
        )
    
    data['Momentum_Weight'] = (abs(data['Momentum_Asymmetry']) * 
                              count_sign_consistency(data['close'] / data['close'].shift(1) - 1, 3))
    data['Volume_Weight'] = abs(data['Net_Volume_Pressure']) * data['Range_Persistence']
    data['Breakout_Weight'] = abs(data['Breakout_Asymmetry']) * abs(data['Fractal_Strength'])
    data['Sync_Weight'] = abs(data['Synchronization_Strength']) * data['Amount_Volume_Ratio']
    
    # Final Alpha Assembly
    data['Weighted_Sum'] = (data['Momentum_Core'] * data['Momentum_Weight'] + 
                           data['Volume_Core'] * data['Volume_Weight'] + 
                           data['Breakout_Core'] * data['Breakout_Weight'] + 
                           data['Sync_Core'] * data['Sync_Weight'])
    
    data['Weight_Sum'] = (data['Momentum_Weight'] + data['Volume_Weight'] + 
                         data['Breakout_Weight'] + data['Sync_Weight'])
    
    data['Base_Alpha'] = data['Weighted_Sum'] / data['Weight_Sum'].replace(0, np.nan)
    
    # Enhanced Alpha
    data['Enhanced_Alpha'] = (data['Base_Alpha'] * (1 + data['Volume_Price_Momentum']) * 
                             (1 + data['Fractal_Momentum_Integration']))
    
    # Return the final alpha factor
    return data['Enhanced_Alpha']
