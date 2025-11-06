import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate VWAP
    df['VWAP'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    # Calculate Fracture Magnitude (placeholder - using volatility-based proxy)
    df['Fracture_Magnitude'] = (df['high'] - df['low']).rolling(window=5).std().fillna(0)
    
    # Fracture Momentum Integration
    df['Fracture_Weighted_Momentum'] = ((df['close'] - df['close'].shift(3)) * 
                                       df['Fracture_Magnitude'] * 
                                       (df['VWAP'] - df['VWAP'].shift(1)) * 
                                       np.sign(df['close'] - df['VWAP']))
    
    df['Microstructure_Fracture_Divergence'] = ((df['close'] - df['VWAP']) / (df['high'] - df['low'] + 1e-8) * 
                                               (df['VWAP'] - df['VWAP'].shift(1)) * 
                                               df['Fracture_Magnitude'] * 
                                               np.sign(df['close'] - df['close'].shift(1)))
    
    df['Volume_Fracture_Momentum'] = ((df['volume'] / df['volume'].shift(1)) * 
                                     (df['VWAP'] - df['VWAP'].shift(1)) * 
                                     df['Fracture_Magnitude'] * 
                                     np.sign(df['close'] - df['VWAP']))
    
    # Fracture Volatility Dynamics
    df['Intraday_Volatility_Fractures'] = ((df['high'] - df['low']) / (df['open'] - df['close'].shift(1) + 1e-8) * 
                                          df['volume'] * df['Fracture_Magnitude'])
    
    df['Microstructure_Volatility_Fracture'] = (((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
                                               (df['VWAP'] - df['VWAP'].shift(1)) * 
                                               df['Fracture_Magnitude'] * 
                                               np.sign(df['close'] - df['VWAP'])))
    
    df['Fracture_Range_Dynamics'] = ((((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) - 
                                     ((df['high'].shift(1) - df['low'].shift(1)) / (df['high'].shift(2) - df['low'].shift(2) + 1e-8))) * 
                                    (df['close'] - df['VWAP']) / (df['high'] - df['low'] + 1e-8) * 
                                    df['Fracture_Magnitude']))
    
    # Fracture-Flow Asymmetry
    df['Fracture_Adaptive_Flow'] = (((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * 
                                   (df['amount'] / df['amount'].shift(1) - 1) * 
                                   df['Fracture_Magnitude']))
    
    df['Microstructure_Absorption_Fracture'] = (((df['close'] - df['VWAP']) / (df['high'] - df['low'] + 1e-8) * 
                                               (df['volume'] / df['volume'].shift(1) - 1) * 
                                               df['Fracture_Magnitude'] * 
                                               np.sign(df['close'] - df['VWAP'])))
    
    df['Volume_Fracture_Alignment'] = (np.sign(df['close'] - df['open']) * 
                                     np.sign(df['volume'] - df['volume'].shift(1)) * 
                                     (df['VWAP'] - df['VWAP'].shift(1)) * 
                                     df['Fracture_Magnitude'])
    
    # Fracture Pattern Recognition
    df['Fracture_Range_Ratio'] = ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
                                 df['Fracture_Magnitude'])
    
    df['Fracture_Range_Breakout'] = df['Fracture_Range_Ratio'] - 1
    
    df['Volume_Fracture_Ratio'] = (df['volume'] / df['volume'].shift(1)) * df['Fracture_Magnitude']
    df['Volume_Fracture_Surge'] = (df['volume'] / df['volume'].shift(1) - 1) * df['Fracture_Magnitude']
    
    df['Price_Volume_Fracture_Congestion'] = ((1 - df['Fracture_Range_Ratio']) * 
                                            (1 - np.abs(df['Volume_Fracture_Ratio'] - 1)))
    
    df['Fracture_Breakout_Validation'] = (np.sign(df['close'] - df['open']) * 
                                        df['Fracture_Range_Breakout'] * 
                                        df['Volume_Fracture_Surge'])
    
    # Asymmetric Fracture Dynamics
    df['Upward_Fracture_Asymmetry'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) - 
                                     (df['volume'] / df['volume'].shift(1) - 1))
    
    df['Downward_Fracture_Asymmetry'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) - 
                                       (df['volume'] / df['volume'].shift(1) - 1))
    
    df['Gap_Fracture_Divergence'] = ((df['open'] - df['close'].shift(1)) * 
                                   df['volume'] / (df['high'] - df['low'] + 1e-8) * 
                                   np.sign(df['open'] - df['close'].shift(1)))
    
    # Fracture Pressure Dynamics
    df['Fracture_Up_Tick_Pressure'] = np.where((df['close'] > df['open']) & (df['Fracture_Magnitude'] > 0.5),
                                             (df['close'] - df['open']) * df['volume'], 0)
    
    df['Fracture_Down_Tick_Pressure'] = np.where((df['close'] < df['open']) & (df['Fracture_Magnitude'] > 0.5),
                                               (df['open'] - df['close']) * df['volume'], 0)
    
    df['Fracture_Asymmetry_Magnitude'] = (np.abs(df['Upward_Fracture_Asymmetry'] - df['Downward_Fracture_Asymmetry']) * 
                                        df['volume'] * df['Fracture_Magnitude'])
    
    df['Fracture_Pressure_Momentum'] = ((df['Fracture_Up_Tick_Pressure'] - df['Fracture_Down_Tick_Pressure']) - 
                                      (df['Fracture_Up_Tick_Pressure'].shift(1) - df['Fracture_Down_Tick_Pressure'].shift(1)))
    
    # Regime-Adaptive Fracture Components
    df['High_Fracture_Breakout'] = (((df['high'] - df['low']) > (df['high'].shift(1) - df['low'].shift(1)) * 1.3) & 
                                  (df['Fracture_Magnitude'] > 0.8)).astype(int)
    
    df['High_Fracture_Reversal'] = (((df['high'] - df['low']) < (df['high'].shift(1) - df['low'].shift(1)) * 0.8) & 
                                  (df['Fracture_Magnitude'] > 0.8)).astype(int)
    
    df['Low_Fracture_Accumulation'] = (((df['high'] - df['low']) < (df['high'].shift(1) - df['low'].shift(1)) * 0.7) & 
                                     (df['Fracture_Magnitude'] < 0.2)).astype(int)
    
    df['Low_Fracture_Compression'] = (df['Low_Fracture_Accumulation'] * 
                                    df['Fracture_Range_Ratio'] * 
                                    df['Fracture_Pressure_Momentum'])
    
    df['Fracture_Regime_Shift'] = ((~df['High_Fracture_Breakout'].astype(bool)) & 
                                 (~df['Low_Fracture_Accumulation'].astype(bool))).astype(int)
    
    df['Fracture_Transition_Momentum'] = (df['Microstructure_Volatility_Fracture'] * 
                                        df['Volume_Fracture_Alignment'])
    
    # Multi-Timeframe Fracture Integration
    df['Intraday_Fracture_Asymmetry'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) - 
                                       (df['volume'] / df['volume'].shift(1) - 1))
    
    df['Closing_Fracture_Divergence'] = ((df['close'] - (df['high'] + df['low'])/2) * 
                                       df['volume'] / (df['high'] - df['low'] + 1e-8) * 
                                       np.sign(df['close'] - df['open']))
    
    df['Three_Day_Fracture_Trend'] = ((df['close'] - df['close'].shift(3)) / 3 * 
                                    df['Fracture_Magnitude'])
    
    # Fracture Efficiency Enhancement
    df['Fracture_Daily_Efficiency'] = (np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * 
                                     df['Fracture_Magnitude'])
    
    df['Fracture_Efficiency_Change'] = (df['Fracture_Daily_Efficiency'] - 
                                      df['Fracture_Daily_Efficiency'].rolling(window=5).mean())
    
    df['Volume_Fracture_Clustering'] = df['Volume_Fracture_Ratio'] - df['Volume_Fracture_Ratio'].shift(1)
    
    df['Price_Volume_Fracture_Alignment'] = (np.sign(df['close'] - df['open']) * 
                                           np.sign(df['volume'] - df['volume'].shift(1)) * 
                                           np.sign(df['open'] - df['close'].shift(1)))
    
    df['Fracture_Efficiency_Momentum'] = (df['Fracture_Efficiency_Change'] * 
                                        df['Volume_Fracture_Clustering'])
    
    # Dynamic Fracture Alpha Construction
    df['Volatility_Fracture_Breakout_Alpha'] = (df['High_Fracture_Breakout'] * 
                                              df['Fracture_Asymmetry_Magnitude'] * 
                                              df['Intraday_Fracture_Asymmetry'])
    
    df['Volatility_Fracture_Reversal_Alpha'] = (df['High_Fracture_Reversal'] * 
                                              np.sign(df['Gap_Fracture_Divergence']) * 
                                              np.sign(df['Upward_Fracture_Asymmetry'] - df['Downward_Fracture_Asymmetry']))
    
    df['Accumulation_Fracture_Alpha'] = (df['Low_Fracture_Accumulation'] * 
                                       df['Fracture_Pressure_Momentum'] * 
                                       (1 - np.abs(df['Volume_Fracture_Ratio'] - 1)))
    
    df['Compression_Breakout_Alpha'] = (df['Low_Fracture_Compression'] * 
                                      df['Fracture_Breakout_Validation'] * 
                                      df['High_Fracture_Breakout'])
    
    df['Regime_Shift_Fracture_Alpha'] = (df['Fracture_Regime_Shift'] * 
                                       df['Fracture_Transition_Momentum'] * 
                                       df['Price_Volume_Fracture_Congestion'])
    
    df['Asymmetry_Transition_Alpha'] = (df['Regime_Shift_Fracture_Alpha'] * 
                                      df['Gap_Fracture_Divergence'] * 
                                      df['Closing_Fracture_Divergence'])
    
    # Final Fracture Alpha Synthesis
    df['High_Fracture_Weight'] = (df['High_Fracture_Breakout'] * 
                                (df['Volatility_Fracture_Breakout_Alpha'] - df['Volatility_Fracture_Reversal_Alpha']))
    
    df['Low_Fracture_Weight'] = (df['Low_Fracture_Accumulation'] * 
                               df['Accumulation_Fracture_Alpha'] * 
                               df['Compression_Breakout_Alpha'])
    
    df['Transition_Weight'] = (df['Fracture_Regime_Shift'] * 
                             df['Asymmetry_Transition_Alpha'])
    
    df['Volume_Fracture_Confirmation'] = (df['Intraday_Fracture_Asymmetry'] * 
                                        df['volume'] / df['volume'].shift(1))
    
    df['Pattern_Fracture_Confirmation'] = (df['Intraday_Fracture_Asymmetry'].rolling(window=3).apply(
        lambda x: np.sum(np.sign(x) == np.sign(df.loc[x.index, 'Three_Day_Fracture_Trend'])), raw=False) * 
        df['Intraday_Fracture_Asymmetry'])
    
    # Final Alpha Calculation
    df['Core_Fracture_Alpha'] = ((df['High_Fracture_Weight'] + df['Low_Fracture_Weight'] + df['Transition_Weight']) * 
                               df['Volume_Fracture_Confirmation'])
    
    df['Refined_Fracture_Alpha'] = (df['Core_Fracture_Alpha'] * 
                                  df['Pattern_Fracture_Confirmation'] * 
                                  df['Fracture_Efficiency_Momentum'])
    
    # Return the final alpha factor
    return df['Refined_Fracture_Alpha'].fillna(0)
