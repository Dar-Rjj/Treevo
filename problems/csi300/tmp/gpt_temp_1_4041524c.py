import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper function for rolling percentiles
    def rolling_percentile(series, window, percentile):
        return series.rolling(window).apply(lambda x: np.percentile(x, percentile), raw=True)
    
    # Calculate True Range
    data['TrueRange'] = np.maximum(data['high'] - data['low'], 
                                  np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                            abs(data['low'] - data['close'].shift(1))))
    
    # Fractal Asymmetric Momentum Framework
    # Multi-Scale Asymmetric Momentum
    data['Micro_Asymmetric_Momentum'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    data['High_2'] = data['high'].rolling(window=3).max()  # t-2 to t
    data['Low_2'] = data['low'].rolling(window=3).min()
    data['Meso_Asymmetric_Momentum'] = (data['High_2'] - data['open']) - (data['open'] - data['Low_2'])
    
    data['High_5'] = data['high'].rolling(window=6).max()  # t-5 to t
    data['Low_5'] = data['low'].rolling(window=6).min()
    data['Macro_Asymmetric_Momentum'] = (data['High_5'] - data['open']) - (data['open'] - data['Low_5'])
    
    data['Fractal_Asymmetric_Cascade'] = (data['Micro_Asymmetric_Momentum'] * 
                                         data['Meso_Asymmetric_Momentum'] * 
                                         data['Macro_Asymmetric_Momentum'])
    
    # Volume Fractal Asymmetry
    data['Volume_Micro_Asymmetry'] = ((data['high'] - data['open']) / (data['volume'] + 0.001)) - \
                                    ((data['close'] - data['low']) / (data['volume'] + 0.001))
    
    data['Volume_Meso_Asymmetry'] = ((data['High_2'] - data['open']) / (data['volume'] + 0.001)) - \
                                   ((data['close'] - data['Low_2']) / (data['volume'] + 0.001))
    
    data['Volume_Macro_Asymmetry'] = ((data['High_5'] - data['open']) / (data['volume'] + 0.001)) - \
                                    ((data['close'] - data['Low_5']) / (data['volume'] + 0.001))
    
    data['Volume_Fractal_Asymmetric_Cascade'] = (data['Volume_Micro_Asymmetry'] * 
                                                data['Volume_Meso_Asymmetry'] * 
                                                data['Volume_Macro_Asymmetry'])
    
    # Fractal Asymmetric Synchronization
    data['Price_Volume_Fractal_Alignment'] = np.sign(data['Fractal_Asymmetric_Cascade']) * \
                                            np.sign(data['Volume_Fractal_Asymmetric_Cascade'])
    
    data['Fractal_Asymmetric_Divergence'] = abs(data['Fractal_Asymmetric_Cascade'] - 
                                               data['Volume_Fractal_Asymmetric_Cascade'])
    
    data['Synchronized_Fractal_Asymmetry'] = (data['Fractal_Asymmetric_Cascade'] * 
                                             data['Volume_Fractal_Asymmetric_Cascade'] * 
                                             (1 - data['Fractal_Asymmetric_Divergence']))
    
    # Efficiency-Compression Fractal Dynamics
    # Fractal Range Efficiency
    data['Micro_Range_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['Meso_Range_Efficiency'] = (data['close'] - data['open']) / (data['High_2'] - data['Low_2'] + 0.001)
    data['Macro_Range_Efficiency'] = (data['close'] - data['open']) / (data['High_5'] - data['Low_5'] + 0.001)
    
    data['Fractal_Efficiency_Cascade'] = (data['Micro_Range_Efficiency'] * 
                                        data['Meso_Range_Efficiency'] * 
                                        data['Macro_Range_Efficiency'])
    
    # Fractal Volatility Compression
    data['Micro_Compression'] = np.where(data['TrueRange'] < rolling_percentile(data['TrueRange'], 6, 30), 1,
                                       np.where(data['TrueRange'] > rolling_percentile(data['TrueRange'], 6, 70), -1, 0))
    
    data['Meso_Compression'] = np.where(data['TrueRange'] < rolling_percentile(data['TrueRange'], 11, 30), 1,
                                      np.where(data['TrueRange'] > rolling_percentile(data['TrueRange'], 11, 70), -1, 0))
    
    data['Macro_Compression'] = np.where(data['TrueRange'] < rolling_percentile(data['TrueRange'], 21, 30), 1,
                                       np.where(data['TrueRange'] > rolling_percentile(data['TrueRange'], 21, 70), -1, 0))
    
    data['Fractal_Compression_Cascade'] = (data['Micro_Compression'] - data['Meso_Compression']) * \
                                         (data['Meso_Compression'] - data['Macro_Compression'])
    
    # Fractal Efficiency-Compression Integration
    data['Efficiency_Compression_Micro'] = (data['Micro_Compression'] - data['Meso_Compression']) * \
                                          data['Micro_Range_Efficiency']
    
    data['Efficiency_Compression_Meso'] = (data['Meso_Compression'] - data['Macro_Compression']) * \
                                         data['Meso_Range_Efficiency']
    
    data['Fractal_Efficiency_Compression_Core'] = data['Efficiency_Compression_Micro'] * \
                                                 data['Efficiency_Compression_Meso']
    
    # Regime Momentum Asymmetric Persistence
    # Asymmetric Regime Persistence
    asymmetric_condition = (data['high'] - data['open']) > (data['open'] - data['low'])
    
    data['Short_Asymmetric_Persistence'] = asymmetric_condition.rolling(window=2).sum() - \
                                          (~asymmetric_condition).rolling(window=2).sum()
    
    data['Medium_Asymmetric_Persistence'] = asymmetric_condition.rolling(window=5).sum() - \
                                           (~asymmetric_condition).rolling(window=5).sum()
    
    data['Long_Asymmetric_Persistence'] = asymmetric_condition.rolling(window=13).sum() - \
                                         (~asymmetric_condition).rolling(window=13).sum()
    
    data['Asymmetric_Regime_Persistence_Cascade'] = (data['Short_Asymmetric_Persistence'] * 
                                                    data['Medium_Asymmetric_Persistence'] * 
                                                    data['Long_Asymmetric_Persistence'])
    
    # Volume Asymmetric Persistence
    volume_short_condition = data['volume'] > data['volume'].shift(1)
    volume_medium_condition = data['volume'] > data['volume'].shift(3)
    volume_long_condition = data['volume'] > data['volume'].shift(8)
    
    data['Volume_Short_Asymmetric'] = (volume_short_condition.rolling(window=2).sum() - 
                                      (~volume_short_condition).rolling(window=2).sum()) * \
                                     np.sign(data['Micro_Asymmetric_Momentum'])
    
    data['Volume_Medium_Asymmetric'] = (volume_medium_condition.rolling(window=5).sum() - 
                                       (~volume_medium_condition).rolling(window=5).sum()) * \
                                      np.sign(data['Micro_Asymmetric_Momentum'])
    
    data['Volume_Long_Asymmetric'] = (volume_long_condition.rolling(window=13).sum() - 
                                     (~volume_long_condition).rolling(window=13).sum()) * \
                                    np.sign(data['Micro_Asymmetric_Momentum'])
    
    data['Volume_Asymmetric_Persistence_Cascade'] = (data['Volume_Short_Asymmetric'] * 
                                                    data['Volume_Medium_Asymmetric'] * 
                                                    data['Volume_Long_Asymmetric'])
    
    # Asymmetric Regime Synchronization
    data['Price_Volume_Asymmetric_Alignment'] = np.sign(data['Asymmetric_Regime_Persistence_Cascade']) * \
                                               np.sign(data['Volume_Asymmetric_Persistence_Cascade'])
    
    data['Asymmetric_Regime_Divergence'] = abs(data['Asymmetric_Regime_Persistence_Cascade'] - 
                                              data['Volume_Asymmetric_Persistence_Cascade'])
    
    data['Synchronized_Asymmetric_Regime'] = (data['Asymmetric_Regime_Persistence_Cascade'] * 
                                             data['Volume_Asymmetric_Persistence_Cascade'] * 
                                             (1 - data['Asymmetric_Regime_Divergence']))
    
    # Range-Adaptive Asymmetric Momentum
    # Fractal Range Context
    data['Micro_Range'] = data['high'] - data['low']
    data['Meso_Range'] = data['High_2'] - data['Low_2']
    data['Macro_Range'] = data['High_5'] - data['Low_5']
    
    data['Fractal_Range_Stability'] = (data['Meso_Range'] / (data['Micro_Range'] + 0.001)) * \
                                     (data['Macro_Range'] / (data['Meso_Range'] + 0.001))
    
    # Asymmetric Range Momentum
    data['Micro_Asymmetric_Range_Momentum'] = data['Micro_Asymmetric_Momentum'] / (data['Micro_Range'] + 0.001)
    data['Meso_Asymmetric_Range_Momentum'] = data['Meso_Asymmetric_Momentum'] / (data['Meso_Range'] + 0.001)
    data['Macro_Asymmetric_Range_Momentum'] = data['Macro_Asymmetric_Momentum'] / (data['Macro_Range'] + 0.001)
    
    data['Fractal_Asymmetric_Range_Cascade'] = (data['Micro_Asymmetric_Range_Momentum'] * 
                                               data['Meso_Asymmetric_Range_Momentum'] * 
                                               data['Macro_Asymmetric_Range_Momentum'])
    
    # Range-Enhanced Asymmetric Integration
    data['Range_Weighted_Asymmetric'] = data['Fractal_Asymmetric_Range_Cascade'] * data['Fractal_Range_Stability']
    data['Asymmetric_Range_Divergence'] = np.sign(data['Micro_Asymmetric_Range_Momentum']) * \
                                         (1 - np.sign(data['Meso_Asymmetric_Range_Momentum']))
    data['Enhanced_Asymmetric_Range'] = data['Range_Weighted_Asymmetric'] * data['Asymmetric_Range_Divergence']
    
    # Composite Alpha Generation
    # Core Fractal Asymmetric Components
    data['Fractal_Asymmetric_Core'] = data['Synchronized_Fractal_Asymmetry'] * data['Fractal_Asymmetric_Cascade']
    data['Asymmetric_Regime_Core'] = data['Synchronized_Asymmetric_Regime'] * data['Asymmetric_Regime_Persistence_Cascade']
    data['Efficiency_Compression_Core'] = data['Fractal_Efficiency_Compression_Core'] * data['Fractal_Efficiency_Cascade']
    
    # Range-Enhanced Framework
    data['Range_Enhanced_Asymmetric'] = data['Enhanced_Asymmetric_Range'] * data['Fractal_Asymmetric_Range_Cascade']
    data['Asymmetric_Range_Alignment'] = data['Asymmetric_Range_Divergence'] * np.sign(data['Range_Enhanced_Asymmetric'])
    data['Integrated_Asymmetric_Range'] = data['Range_Enhanced_Asymmetric'] * data['Asymmetric_Range_Alignment']
    
    # Multi-Scale Integration
    data['Base_Asymmetric_Factor'] = (data['Fractal_Asymmetric_Core'] * 
                                     data['Asymmetric_Regime_Core'] * 
                                     data['Efficiency_Compression_Core'])
    
    data['Range_Multiplier'] = data['Base_Asymmetric_Factor'] * data['Integrated_Asymmetric_Range']
    
    data['Scale_Weighted_Asymmetric'] = (data['Range_Multiplier'] * 
                                        data['Micro_Asymmetric_Momentum'] * 
                                        data['Meso_Asymmetric_Momentum'] * 
                                        data['Macro_Asymmetric_Momentum'])
    
    # Final Alpha Composition
    data['Alpha_Base'] = data['Scale_Weighted_Asymmetric'] * data['Fractal_Asymmetric_Cascade']
    data['Final_Alpha'] = data['Alpha_Base'] * data['Synchronized_Fractal_Asymmetry']
    
    return data['Final_Alpha']
