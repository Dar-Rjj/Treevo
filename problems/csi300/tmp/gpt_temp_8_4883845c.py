import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Discovery Asymmetry
    df['Opening_Pressure_Asymmetry'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Intraday_Range_Asymmetry'] = (df['high'] - df['low']) / df['close']
    df['Closing_Efficiency_Asymmetry'] = (df['close'] - (df['high'] + df['low'])/2) / (df['high'] - df['low'])
    
    # Volume Microstructure Asymmetry
    df['Volume_MA_5'] = df['volume'].rolling(window=5).mean()
    df['Volume_Magnitude_Asymmetry'] = df['volume'] / df['Volume_MA_5']
    df['Volume_Price_Pressure'] = (df['close'] - df['open']) * df['volume'] / df['Volume_MA_5']
    df['Volume_Clustering_Asymmetry'] = df['volume'].rolling(window=5).max() / df['volume'].rolling(window=5).min()
    
    # Momentum Phase Analysis
    df['Short_term_Momentum'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['Medium_term_Momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['Momentum_Phase_Alignment'] = np.sign(df['Short_term_Momentum']) * np.sign(df['Medium_term_Momentum'])
    
    # Volatility-Regime Microstructure
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['close']
    df['Volatility_Trend'] = df['close'].rolling(window=5).std() / df['close'].shift(5).rolling(window=5).std()
    df['Volatility_Adjusted_Opening'] = df['Opening_Pressure_Asymmetry'] / df['Intraday_Volatility']
    
    # Order Flow Asymmetry
    df['Close_Change'] = df['close'] > df['close'].shift(1)
    df['Up_Tick_Volume'] = df['volume'].rolling(window=5).apply(lambda x: np.sum(x[1:][df['Close_Change'].shift(-1).rolling(window=4).mean().values[:4] > 0]), raw=False)
    df['Down_Tick_Volume'] = df['volume'].rolling(window=5).apply(lambda x: np.sum(x[1:][df['Close_Change'].shift(-1).rolling(window=4).mean().values[:4] < 0]), raw=False)
    df['Net_Pressure_Asymmetry'] = (df['Up_Tick_Volume'] - df['Down_Tick_Volume']) / (df['Up_Tick_Volume'] + df['Down_Tick_Volume'])
    
    # Fractal Efficiency Metrics
    df['Range_Efficiency'] = (df['close'] - df['close'].shift(4)) / (df['high'].rolling(window=5).sum() - df['low'].rolling(window=5).sum())
    df['Directional_Consistency'] = np.sign(df['close'] - df['close'].shift(1)) * np.sign(df['close'].shift(1) - df['close'].shift(2))
    df['Volume_Stability'] = df['volume'].rolling(window=5).std() / df['volume'].rolling(window=5).mean()
    
    # Composite Alpha Synthesis
    df['Microstructure_Momentum'] = df['Opening_Pressure_Asymmetry'] * df['Volume_Price_Pressure'] * df['Momentum_Phase_Alignment']
    df['Volatility_Volume_Alignment'] = df['Volatility_Adjusted_Opening'] * df['Volume_Magnitude_Asymmetry'] * df['Net_Pressure_Asymmetry']
    df['Efficiency_Enhanced_Momentum'] = df['Range_Efficiency'] * df['Directional_Consistency'] * df['Short_term_Momentum']
    
    # Multi-Scale Refinement
    df['Volatility_Regime_Adaptation'] = df['Microstructure_Momentum'] * df['Volatility_Trend']
    df['Volume_Breakout_Alignment'] = df['Efficiency_Enhanced_Momentum'] * (df['volume'] / df['volume'].shift(5).rolling(window=5).mean())
    df['Pressure_Trend_Confirmation'] = df['Microstructure_Momentum'] * (df['Net_Pressure_Asymmetry'] - df['Net_Pressure_Asymmetry'].shift(1))
    df['Clustering_Adjustment'] = df['Volatility_Volume_Alignment'] / df['Volume_Clustering_Asymmetry']
    
    # Final Alpha Integration
    df['Asymmetric_Microstructure_Factor'] = df['Microstructure_Momentum'] * df['Volatility_Volume_Alignment']
    df['Efficiency_Momentum_Factor'] = df['Efficiency_Enhanced_Momentum'] * df['Volume_Breakout_Alignment']
    df['Regime_Adaptive_Alpha'] = df['Asymmetric_Microstructure_Factor'] * df['Volatility_Regime_Adaptation']
    
    # Final alpha factor
    alpha = df['Regime_Adaptive_Alpha']
    
    # Clean up intermediate columns
    cols_to_drop = ['Opening_Pressure_Asymmetry', 'Intraday_Range_Asymmetry', 'Closing_Efficiency_Asymmetry',
                   'Volume_MA_5', 'Volume_Magnitude_Asymmetry', 'Volume_Price_Pressure', 'Volume_Clustering_Asymmetry',
                   'Short_term_Momentum', 'Medium_term_Momentum', 'Momentum_Phase_Alignment', 'Intraday_Volatility',
                   'Volatility_Trend', 'Volatility_Adjusted_Opening', 'Close_Change', 'Up_Tick_Volume', 'Down_Tick_Volume',
                   'Net_Pressure_Asymmetry', 'Range_Efficiency', 'Directional_Consistency', 'Volume_Stability',
                   'Microstructure_Momentum', 'Volatility_Volume_Alignment', 'Efficiency_Enhanced_Momentum',
                   'Volatility_Regime_Adaptation', 'Volume_Breakout_Alignment', 'Pressure_Trend_Confirmation',
                   'Clustering_Adjustment', 'Asymmetric_Microstructure_Factor', 'Efficiency_Momentum_Factor']
    
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return alpha
