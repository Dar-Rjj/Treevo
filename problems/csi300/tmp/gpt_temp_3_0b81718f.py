import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Quantum Volatility-Liquidity Integration
    df['Volatility_Quantum_Efficiency'] = ((df['open'] / df['close'].shift(1) - 1) * 
                                         (df['high'] - df['low']) / df['volume'])
    
    df['Liquidity_Quantum_Density'] = (df['amount'] / (df['high'] - df['low']) * 
                                     df['volume'] / df['volume'].shift(1))
    
    df['Amount_Mean_4'] = df['amount'].rolling(window=4).mean()
    df['Quantum_Range_Utilization'] = ((df['close'] - df['low']) / (df['high'] - df['low']) * 
                                     df['amount'] / df['Amount_Mean_4'])
    
    # Multi-Scale Momentum Decay Patterns
    df['Price_Momentum_Decay'] = ((df['close'] - df['close'].shift(5)) * 
                                (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)))
    
    df['Volume_Momentum_Decay'] = (df['volume'] / df['volume'].shift(3) * 
                                 (df['amount'] / df['volume']) / 
                                 (df['amount'].shift(3) / df['volume'].shift(3)))
    
    df['High_Low_Range_Mean_4'] = (df['high'] - df['low']).rolling(window=4).mean()
    df['Volatility_Momentum_Decay'] = ((df['high'] - df['low']) / df['High_Low_Range_Mean_4'] * 
                                     df['volume'] / df['volume'].shift(1))
    
    # Quantum Absorption-Volatility Asymmetry
    df['Volume_Absorption_Quantum'] = (df['volume'] * df['volume'].shift(1) / 
                                     (df['volume'] + df['volume'].shift(1)) * 
                                     abs(df['close'] - df['open']))
    
    # Define up/down days
    df['is_up_day'] = (df['close'] > df['open']).astype(int)
    df['is_down_day'] = (df['close'] < df['open']).astype(int)
    
    df['Up_Day_Volume'] = df['volume'] * df['is_up_day']
    df['Down_Day_Volume'] = df['volume'] * df['is_down_day']
    df['Up_Day_Range'] = (df['high'] - df['low']) * df['is_up_day']
    df['Down_Day_Range'] = (df['high'] - df['low']) * df['is_down_day']
    
    df['Volatility_Asymmetry_Quantum'] = ((df['Up_Day_Volume'] * df['Up_Day_Range']) / 
                                        (df['Down_Day_Volume'] * df['Down_Day_Range'] + 0.0001) * 
                                        df['amount'])
    
    df['Quantum_Asymmetry_Ratio'] = (df['Volume_Absorption_Quantum'] / 
                                   (df['Volatility_Asymmetry_Quantum'] + 0.0001))
    
    # Microstructure Decay Convergence
    df['Price_Volume_Decay_Alignment'] = (np.sign(df['Price_Momentum_Decay'] - 1) * 
                                        np.sign(df['Volume_Momentum_Decay'] - 1))
    
    df['Volatility_Decay_Quantum'] = (df['Volatility_Momentum_Decay'] * 
                                    df['Quantum_Range_Utilization'])
    
    df['Microstructure_Convergence'] = (df['Price_Volume_Decay_Alignment'] * 
                                      df['Volatility_Decay_Quantum'])
    
    # Quantum Regime-Adaptive Signals
    df['High_Volatility_Quantum'] = (-df['Quantum_Asymmetry_Ratio'] * 
                                   abs(df['Price_Momentum_Decay']))
    
    df['Trending_Quantum'] = (df['Volatility_Quantum_Efficiency'] * 
                            df['Volume_Momentum_Decay'] * 
                            df['Quantum_Asymmetry_Ratio'])
    
    df['Compressed_Range_Quantum'] = (df['Quantum_Range_Utilization'] * 
                                    df['Liquidity_Quantum_Density'])
    
    # Quantum Momentum-Volatility Entanglement
    df['Momentum_Volatility_Quantum'] = ((df['close'] / df['close'].shift(5) - 1) * 
                                       (df['high'] - df['low']) / df['volume'])
    
    # Smart Money Quantum Flow
    df['Smart_Money_Quantum_Flow'] = 0
    for i in range(4, len(df)):
        window_data = df.iloc[i-4:i+1]
        smart_money = np.sum(window_data['amount'] * 
                           (window_data['close'] - window_data['open']) * 
                           window_data['volume'])
        df.iloc[i, df.columns.get_loc('Smart_Money_Quantum_Flow')] = np.sign(smart_money)
    
    df['Transaction_Quantum_Efficiency'] = (df['amount'] * abs(df['close'] - df['open']) / 
                                          (df['high'] - df['low']) * df['volume'])
    
    # Quantum Enhancement Dynamics
    df['Quantum_Jump_Detection'] = (abs(df['Liquidity_Quantum_Density'] - 
                                      df['Liquidity_Quantum_Density'].shift(1)) * 
                                  df['volume'])
    
    df['Decay_Quantum_Acceleration'] = (df['Price_Momentum_Decay'] / 
                                      df['Volume_Momentum_Decay'])
    
    df['Quantum_Consistency'] = (np.sign(df['Microstructure_Convergence']) * 
                               np.sign(df['Quantum_Asymmetry_Ratio']))
    
    # Final Quantum Microstructure Alpha
    # Quantum State Integration
    df['Short_term_Quantum'] = (df['Volatility_Quantum_Efficiency'] * 
                              df['Price_Momentum_Decay'])
    
    df['Medium_term_Quantum'] = (df['Liquidity_Quantum_Density'] * 
                               df['Volume_Momentum_Decay'])
    
    df['Scaled_Quantum_Signal'] = (df['Microstructure_Convergence'] * 
                                 df['Short_term_Quantum'] * 
                                 df['Medium_term_Quantum'])
    
    # Quantum Coherence Enhancement
    df['Temporal_Quantum_Coherence'] = (df['Price_Volume_Decay_Alignment'] * 
                                      df['Volume_Absorption_Quantum'])
    
    df['Spatial_Quantum_Coherence'] = (df['Quantum_Asymmetry_Ratio'] * 
                                     df['Quantum_Range_Utilization'])
    
    df['Coherence_Enhanced_Alpha'] = (df['Scaled_Quantum_Signal'] * 
                                    df['Temporal_Quantum_Coherence'] * 
                                    df['Spatial_Quantum_Coherence'])
    
    # Adaptive Quantum Microstructure Factor
    df['Quantum_Microstructure_Factor'] = (df['Coherence_Enhanced_Alpha'] * 
                                         df['Quantum_Consistency'])
    
    # Clean up intermediate columns
    cols_to_drop = ['Amount_Mean_4', 'High_Low_Range_Mean_4', 'is_up_day', 'is_down_day',
                   'Up_Day_Volume', 'Down_Day_Volume', 'Up_Day_Range', 'Down_Day_Range']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df['Quantum_Microstructure_Factor']
