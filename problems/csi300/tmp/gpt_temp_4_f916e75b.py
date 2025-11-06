import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum & Volume Dynamics
    data['Volume_Weighted_Momentum_Fractal'] = ((data['close'] - data['close'].shift(1)) / (data['volume'] + 0.001)) * \
                                             ((data['close'] - data['close'].shift(2)) / (data['volume'].shift(1) + 0.001)) * \
                                             np.sign(data['close'] - data['close'].shift(1))
    
    data['High_Low_Momentum_Asymmetry'] = ((data['high'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.001)) - \
                                         ((data['close'].shift(1) - data['low']) / (data['close'].shift(1) + 0.001)) * \
                                         ((data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 0.001))
    
    # Volume Regime Persistence
    vol_up_price_up = ((data['volume'] > data['volume'].shift(1)) & (data['close'] > data['close'].shift(1))).astype(int)
    vol_down_price_down = ((data['volume'] < data['volume'].shift(1)) & (data['close'] < data['close'].shift(1))).astype(int)
    
    data['Volume_Regime_Persistence'] = (vol_up_price_up.rolling(window=3, min_periods=1).sum() - 
                                       vol_down_price_down.rolling(window=3, min_periods=1).sum())
    
    # Multi-Scale Momentum Asymmetry
    data['Morning_Momentum'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)) * (data['open'] - data['low'])
    data['Afternoon_Momentum'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)) * (data['high'] - data['close'])
    data['Intraday_Momentum_Asymmetry'] = data['Morning_Momentum'] - data['Afternoon_Momentum']
    
    # Short-Term Momentum Asymmetry
    data['High_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['Low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    
    data['Short_Term_Morning'] = ((data['close'] - data['open']) / (data['High_3d'] - data['Low_3d'] + 0.001)) * (data['open'] - data['Low_3d'])
    data['Short_Term_Afternoon'] = ((data['close'] - data['open']) / (data['High_3d'] - data['Low_3d'] + 0.001)) * (data['High_3d'] - data['close'])
    data['Short_Term_Momentum_Asymmetry'] = data['Short_Term_Morning'] - data['Short_Term_Afternoon']
    
    # Momentum Fractal Cascade
    data['Momentum_Fractal_Cascade'] = data['Intraday_Momentum_Asymmetry'] * data['Short_Term_Momentum_Asymmetry'] * data['Volume_Weighted_Momentum_Fractal']
    data['Momentum_Horizon_Resonance'] = np.sign(data['Short_Term_Momentum_Asymmetry'] - data['Intraday_Momentum_Asymmetry']) * np.sign(data['Volume_Weighted_Momentum_Fractal'])
    
    # Volume-Pressure Asymmetric Dynamics
    data['Buy_Pressure'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['Sell_Pressure'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)
    data['Pressure_Asymmetry'] = (data['Buy_Pressure'] - data['Sell_Pressure']) * np.sign(data['Buy_Pressure'] - data['Sell_Pressure'].shift(1))
    
    data['Morning_Volume_Momentum'] = data['volume'] * data['Morning_Momentum']
    data['Afternoon_Volume_Momentum'] = data['volume'] * data['Afternoon_Momentum']
    data['Volume_Momentum_Differential'] = data['Morning_Volume_Momentum'] - data['Afternoon_Volume_Momentum']
    
    data['Volume_Pressure_Velocity'] = data['Pressure_Asymmetry'] - data['Pressure_Asymmetry'].shift(1)
    data['Volume_Momentum_Alignment'] = np.sign(data['Volume_Momentum_Differential']) * np.sign(data['Volume_Pressure_Velocity']) * data['Pressure_Asymmetry']
    
    # Gap Transmission & Pattern Integration
    data['Micro_Gap'] = (data['open'] - data['close'].shift(1)) * (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    data['High_3d_range'] = data['high'].rolling(window=3, min_periods=1).max()
    data['Low_3d_range'] = data['low'].rolling(window=3, min_periods=1).min()
    data['Meso_Gap'] = (data['open'] - data['close'].shift(3)) * (data['close'] - data['open']) / (data['High_3d_range'] - data['Low_3d_range'] + 0.001)
    
    data['High_8d'] = data['high'].rolling(window=8, min_periods=1).max()
    data['Low_8d'] = data['low'].rolling(window=8, min_periods=1).min()
    data['Macro_Gap'] = (data['open'] - data['close'].shift(8)) * (data['close'] - data['open']) / (data['High_8d'] - data['Low_8d'] + 0.001)
    
    # Reversal Pattern Density
    reversal_patterns = []
    for i in range(len(data)):
        if i < 4:
            reversal_patterns.append(0)
            continue
            
        count = 0
        for j in range(i-4, i):
            if j > 0 and j < len(data)-1:
                if (data['close'].iloc[j] > max(data['close'].iloc[j-1], data['close'].iloc[j+1]) or 
                    data['close'].iloc[j] < min(data['close'].iloc[j-1], data['close'].iloc[j+1])):
                    count += 1
        reversal_patterns.append(count)
    
    data['Reversal_Pattern_Density'] = reversal_patterns * (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 0.001)
    data['Gap_Momentum_Transmission'] = data['Micro_Gap'] * data['Intraday_Momentum_Asymmetry'] * data['Volume_Pressure_Velocity']
    
    # Range Expansion Momentum
    range_expansion = (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))
    range_contraction = (data['high'] - data['low']) < (data['high'].shift(1) - data['low'].shift(1))
    
    data['Range_Expansion_Momentum'] = (range_expansion.rolling(window=3, min_periods=1).sum() - 
                                      range_contraction.rolling(window=3, min_periods=1).sum()) * \
                                     (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.001)
    
    data['Pattern_Momentum_Boost'] = data['Gap_Momentum_Transmission'] * data['Reversal_Pattern_Density'] * data['Range_Expansion_Momentum']
    
    # Amount-Driven Microstructure Dynamics
    data['Momentum_Trade_Size'] = data['amount'] / (data['volume'] + 0.001)
    data['Trade_Size_Momentum_Ratio'] = data['Momentum_Trade_Size'] / (data['amount'].shift(1) / (data['volume'].shift(1) + 0.001) + 0.001)
    data['Trade_Size_Momentum_Volatility'] = np.abs(data['Momentum_Trade_Size'] - (data['amount'].shift(1) / (data['volume'].shift(1) + 0.001))) / \
                                           (data['amount'].shift(1) / (data['volume'].shift(1) + 0.001) + 0.001)
    
    data['Amount_per_Momentum_Unit'] = data['amount'] / (np.abs(data['close'] - data['open']) + 0.001)
    data['Volume_per_Momentum_Unit'] = data['volume'] / (np.abs(data['close'] - data['open']) + 0.001)
    data['Amount_Volume_Momentum_Efficiency'] = data['Amount_per_Momentum_Unit'] * data['Volume_per_Momentum_Unit'] * \
                                              (data['close'] - data['open'])**2 / ((data['open'] - data['close'].shift(1))**2 + (data['close'] - data['open'])**2 + 0.001)
    
    data['Large_Trade_Pressure'] = data['amount'] * (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001) * (data['open'] - data['low'])
    data['Small_Trade_Pressure'] = data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001) * (data['high'] - data['close'])
    data['Trade_Size_Pressure_Divergence'] = data['Large_Trade_Pressure'] - data['Small_Trade_Pressure']
    
    # Volatility-Regime Adaptive Framework
    data['Micro_Volatility'] = (data['high'] - data['low']) / (data['close'].shift(1) + 0.001)
    data['High_2d'] = data['high'].rolling(window=2, min_periods=1).max()
    data['Low_2d'] = data['low'].rolling(window=2, min_periods=1).min()
    data['Meso_Volatility'] = (data['High_2d'] - data['Low_2d']) / (data['close'].shift(1) + 0.001)
    data['High_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['Low_5d'] = data['low'].rolling(window=5, min_periods=1).min()
    data['Macro_Volatility'] = (data['High_5d'] - data['Low_5d']) / (data['close'].shift(1) + 0.001)
    
    data['Micro_Volume_Intensity'] = data['volume'] / (data['volume'].rolling(window=2, min_periods=1).mean() + 0.001)
    data['Meso_Volume_Intensity'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=1).mean() + 0.001)
    data['Macro_Volume_Intensity'] = data['volume'] / (data['volume'].rolling(window=13, min_periods=1).mean() + 0.001)
    
    data['Volatility_Expansion'] = (data['Micro_Volatility'] - data['Meso_Volatility']) * (data['Meso_Volatility'] - data['Macro_Volatility'])
    data['Volume_Expansion'] = (data['Micro_Volume_Intensity'] - data['Meso_Volume_Intensity']) * (data['Meso_Volume_Intensity'] - data['Macro_Volume_Intensity'])
    data['Regime_Transition_Composite'] = data['Volatility_Expansion'] * data['Volume_Expansion']
    
    # Persistence & Divergence Analysis
    price_up = (data['close'] > data['close'].shift(1)).astype(int)
    data['Price_Momentum_Persistence'] = price_up.rolling(window=3, min_periods=1).sum() * np.sign(data['Momentum_Fractal_Cascade'])
    
    volume_up = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['Volume_Momentum_Persistence'] = volume_up.rolling(window=3, min_periods=1).sum() * np.sign(data['Volume_Momentum_Differential'])
    
    vol_comp = (data['Micro_Volatility'] > data['Meso_Volatility']).astype(int) - (data['Micro_Volatility'] < data['Meso_Volatility']).astype(int)
    data['Volatility_Persistence'] = vol_comp.rolling(window=3, min_periods=1).sum()
    
    data['Price_Volume_Momentum_Divergence'] = ((data['close'] - data['close'].shift(2)) / (data['close'].shift(2) + 0.001)) - \
                                             ((data['volume'] - data['volume'].shift(2)) / (data['volume'].shift(2) + 0.001)) * \
                                             np.sign(data['close'] - data['close'].shift(1))
    
    def calc_divergence_strength(row):
        if row['close'] > row['close_prev'] and row['volume'] < row['volume_prev']:
            return 1
        elif row['close'] < row['close_prev'] and row['volume'] > row['volume_prev']:
            return -1
        else:
            return 0
    
    data['close_prev'] = data['close'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['Volatility_Volume_Divergence_Strength'] = data.apply(calc_divergence_strength, axis=1) * \
                                                   (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    
    data['Divergence_Regime_Signal'] = np.sign(data['Price_Volume_Momentum_Divergence']) * \
                                     np.sign(data['Volatility_Volume_Divergence_Strength']) * \
                                     (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 0.001)
    
    data['Momentum_Divergence'] = data['Price_Volume_Momentum_Divergence'] * data['Volatility_Volume_Divergence_Strength'] * data['Divergence_Regime_Signal']
    data['Persistence_Filter'] = (1 + data['Price_Momentum_Persistence']) * (1 + data['Volume_Momentum_Persistence']) * (1 + data['Volatility_Persistence'])
    data['Divergence_Resonance'] = data['Momentum_Divergence'] * data['Persistence_Filter']
    
    # Composite Alpha Construction
    data['Momentum_Core'] = data['Momentum_Fractal_Cascade'] * data['Momentum_Horizon_Resonance'] * data['Volume_Regime_Persistence']
    data['Volume_Core'] = data['Volume_Momentum_Alignment'] * data['Amount_Volume_Momentum_Efficiency'] * data['Pressure_Asymmetry']
    data['Microstructure_Core'] = data['Trade_Size_Pressure_Divergence'] * data['Volume_Momentum_Differential'] * data['High_Low_Momentum_Asymmetry']
    
    data['Base_Alpha'] = data['Momentum_Core'] * data['Volume_Core'] * data['Microstructure_Core']
    data['Enhanced_Alpha'] = data['Base_Alpha'] * data['Divergence_Resonance'] * data['Pattern_Momentum_Boost']
    
    # Final alpha with trend persistence
    intraday_trend_improvement = (data['close'] - data['open']) > (data['close'].shift(1) - data['open'].shift(1))
    data['Multi_Scale_Momentum_Volume_Fractal_Regime_Alpha'] = data['Enhanced_Alpha'] * \
                                                             intraday_trend_improvement.rolling(window=4, min_periods=1).sum() / 4
    
    # Clean up intermediate columns
    final_columns = ['Multi_Scale_Momentum_Volume_Fractal_Regime_Alpha']
    
    return data[final_columns].iloc[:, 0]
