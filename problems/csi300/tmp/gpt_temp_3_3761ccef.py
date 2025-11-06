import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Fracture Components
    data['Intraday_Price_Fracture'] = ((data['high'] - data['close']) / 
                                     (data['close'] - data['low'] + 1e-6) * 
                                     (data['open'] - data['close'].shift(2)) / 
                                     (abs(data['close'] - data['close'].shift(3)) + 1e-6))
    
    data['Gap_Fracture'] = ((data['open'] - data['close'].shift(1)) / 
                           (data['high'].shift(1) - data['low'].shift(1) + 1e-6) * 
                           abs(data['close'] - data['open']) / 
                           (data['high'] - data['low'] + 1e-6))
    
    data['Price_Fracture_Divergence'] = (data['Intraday_Price_Fracture'] - 
                                        data['Gap_Fracture'] * 
                                        np.sign(data['close'] - data['close'].shift(2)))
    
    # Volume Fracture Components
    data['Volume_Intensity_Fracture'] = (data['volume'] / (data['volume'].shift(3) + 1e-6) * 
                                       abs(data['close'] - data['close'].shift(1)) / 
                                       (data['high'] - data['low'] + 1e-6))
    
    # Volume Persistence Fracture
    volume_increase_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            count = 0
            for j in range(i-4, i+1):
                if data['volume'].iloc[j] > data['volume'].iloc[j-1]:
                    count += 1
            volume_increase_count.iloc[i] = count
        else:
            volume_increase_count.iloc[i] = np.nan
    
    data['Volume_Persistence_Fracture'] = (volume_increase_count * data['volume'] / 
                                         data['volume'].rolling(window=5, min_periods=1).sum())
    
    data['Volume_Fracture_Divergence'] = (data['Volume_Intensity_Fracture'] - 
                                         data['Volume_Persistence_Fracture'] * 
                                         np.sign(data['close'] - data['open']))
    
    # Fracture Integration
    data['Price_Volume_Fracture_Correlation'] = (data['Price_Fracture_Divergence'] * 
                                               data['Volume_Fracture_Divergence'])
    
    data['Fracture_Momentum'] = (data['Price_Volume_Fracture_Correlation'] * 
                               (data['close'] - data['close'].shift(2)) / 
                               (data['high'] - data['low'] + 1e-6))
    
    data['Fracture_Acceleration'] = ((data['Fracture_Momentum'] - data['Fracture_Momentum'].shift(3)) / 
                                   (abs(data['Fracture_Momentum'].shift(3)) + 1e-6))
    
    # Multi-Timeframe Fracture Analysis
    data['Recent_Price_Fracture'] = ((data['close'] - data['open']) / 
                                   (data['high'] - data['low'] + 1e-6) * 
                                   (data['high'] - data['close'].shift(1)) / 
                                   (abs(data['close'] - data['close'].shift(1)) + 1e-6))
    
    data['Recent_Volume_Fracture'] = (data['volume'] / (data['volume'].shift(2) + 1e-6) * 
                                    (data['close'] - data['low']) / 
                                    (data['high'] - data['low'] + 1e-6))
    
    data['Short_Term_Fracture_Composite'] = (data['Recent_Price_Fracture'] * 
                                           data['Recent_Volume_Fracture'] * 
                                           np.sign(data['close'] - data['close'].shift(1)))
    
    # Medium-Term Fracture Dynamics
    data['Price_Trend_Fracture'] = ((data['close'] - data['close'].shift(5)) / 
                                  (data['high'].shift(5) - data['low'].shift(5) + 1e-6) * 
                                  abs(data['close'] - data['close'].shift(3)) / 
                                  (data['high'] - data['low'] + 1e-6))
    
    data['Volume_Trend_Fracture'] = (data['volume'].rolling(window=5, min_periods=1).sum() / 
                                   data['volume'].shift(5).rolling(window=5, min_periods=1).sum() * 
                                   (data['close'] - data['open']) / 
                                   (data['high'] - data['low'] + 1e-6))
    
    data['Medium_Term_Fracture_Composite'] = (data['Price_Trend_Fracture'] * 
                                            data['Volume_Trend_Fracture'] * 
                                            np.sign(data['close'] - data['close'].shift(5)))
    
    # Fracture Timeframe Divergence
    data['Short_Medium_Fracture_Gap'] = (data['Short_Term_Fracture_Composite'] - 
                                       data['Medium_Term_Fracture_Composite'])
    
    # Fracture Consistency
    fracture_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 6:
            count = 0
            for j in range(i-6, i+1):
                if np.sign(data['Short_Term_Fracture_Composite'].iloc[j]) == np.sign(data['Medium_Term_Fracture_Composite'].iloc[j]):
                    count += 1
            fracture_consistency.iloc[i] = count
        else:
            fracture_consistency.iloc[i] = np.nan
    
    data['Fracture_Consistency'] = fracture_consistency
    
    data['Timeframe_Fracture_Momentum'] = (data['Short_Medium_Fracture_Gap'] * 
                                         data['Fracture_Consistency'] / 
                                         (abs(data['Short_Term_Fracture_Composite']) + 
                                          abs(data['Medium_Term_Fracture_Composite']) + 1e-6))
    
    # Amount-Enhanced Fracture Framework
    data['Amount_Intensity'] = (data['amount'] / (data['amount'].shift(3) + 1e-6) * 
                              (data['close'] - data['open']) / 
                              (data['high'] - data['low'] + 1e-6))
    
    # Amount Persistence
    amount_increase_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            count = 0
            for j in range(i-4, i+1):
                if data['amount'].iloc[j] > data['amount'].iloc[j-1]:
                    count += 1
            amount_increase_count.iloc[i] = count
        else:
            amount_increase_count.iloc[i] = np.nan
    
    data['Amount_Persistence'] = (amount_increase_count * data['amount'] / 
                                data['amount'].rolling(window=5, min_periods=1).sum())
    
    data['Amount_Fracture'] = (data['Amount_Intensity'] - 
                             data['Amount_Persistence'] * 
                             np.sign(data['close'] - data['close'].shift(2)))
    
    # Amount-Price Fracture Integration
    data['Amount_Price_Fracture_Correlation'] = (data['Amount_Fracture'] * 
                                               data['Price_Fracture_Divergence'])
    
    data['Amount_Volume_Fracture_Correlation'] = (data['Amount_Fracture'] * 
                                                data['Volume_Fracture_Divergence'])
    
    data['Triple_Fracture_Composite'] = (data['Amount_Price_Fracture_Correlation'] * 
                                       data['Amount_Volume_Fracture_Correlation'] * 
                                       data['Price_Volume_Fracture_Correlation'])
    
    # Enhanced Fracture Momentum
    data['Amount_Enhanced_Fracture'] = (data['Triple_Fracture_Composite'] * 
                                      data['amount'] / (data['amount'].shift(5) + 1e-6))
    
    # Fracture Momentum Persistence
    fracture_momentum_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 7:
            count = 0
            for j in range(i-7, i):
                if np.sign(data['Triple_Fracture_Composite'].iloc[j]) == np.sign(data['Triple_Fracture_Composite'].iloc[i]):
                    count += 1
            fracture_momentum_persistence.iloc[i] = count
        else:
            fracture_momentum_persistence.iloc[i] = np.nan
    
    data['Fracture_Momentum_Persistence'] = fracture_momentum_persistence
    
    data['Persistent_Fracture_Momentum'] = (data['Amount_Enhanced_Fracture'] * 
                                          data['Fracture_Momentum_Persistence'] / 7)
    
    # Core Fracture Alpha
    data['Core_Fracture_Alpha'] = (data['Persistent_Fracture_Momentum'] * 
                                 data['Timeframe_Fracture_Momentum'] * 
                                 data['Fracture_Acceleration'])
    
    # Fracture Regime Detection and Modulation
    price_fracture_modulator = pd.Series(1.0, index=data.index)
    volume_fracture_modulator = pd.Series(1.0, index=data.index)
    amount_fracture_modulator = pd.Series(1.0, index=data.index)
    
    for i in range(len(data)):
        # Price Fracture Regimes
        if (abs(data['Price_Fracture_Divergence'].iloc[i]) > 2.5 and 
            abs(data['Intraday_Price_Fracture'].iloc[i]) > 1.8):
            price_fracture_modulator.iloc[i] = 0.4
        elif (abs(data['Price_Fracture_Divergence'].iloc[i]) < 0.3 and 
              abs(data['Intraday_Price_Fracture'].iloc[i]) < 0.4):
            price_fracture_modulator.iloc[i] = 0.15
        
        # Volume Fracture Regimes
        if (data['Volume_Intensity_Fracture'].iloc[i] > 2.2 and 
            data['Volume_Persistence_Fracture'].iloc[i] > 1.5):
            volume_fracture_modulator.iloc[i] = 0.35
        elif (data['Volume_Intensity_Fracture'].iloc[i] < 0.4 and 
              data['Volume_Persistence_Fracture'].iloc[i] < 0.5):
            volume_fracture_modulator.iloc[i] = 0.08
        
        # Amount Fracture Regimes
        if (data['Amount_Intensity'].iloc[i] > 2.0 and 
            data['Amount_Persistence'].iloc[i] > 1.3):
            amount_fracture_modulator.iloc[i] = 0.3
        elif (data['Amount_Intensity'].iloc[i] < 0.35 and 
              data['Amount_Persistence'].iloc[i] < 0.45):
            amount_fracture_modulator.iloc[i] = 0.12
    
    data['Combined_Fracture_Modulator'] = (price_fracture_modulator * 
                                         volume_fracture_modulator * 
                                         amount_fracture_modulator)
    
    # Final Alpha Calculation
    data['Fracture_Intensity_Enhancement'] = (data['Core_Fracture_Alpha'] * 
                                            abs(data['Price_Fracture_Divergence']) / 
                                            (abs(data['Volume_Fracture_Divergence']) + 1e-6))
    
    data['Dynamic_Fracture_Momentum'] = (data['Fracture_Intensity_Enhancement'] * 
                                       data['Combined_Fracture_Modulator'])
    
    data['Final_Alpha'] = (data['Dynamic_Fracture_Momentum'] * 
                         np.sign(data['close'] - data['close'].shift(1)))
    
    return data['Final_Alpha']
