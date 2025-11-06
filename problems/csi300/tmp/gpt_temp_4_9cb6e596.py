import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Dynamics
    data['Intraday_Volatility_Asymmetry'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 0.001) - 
                                            (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001))
    
    # Multi_Timeframe_Volatility_Efficiency
    high_low_range = data['high'] - data['low'] + 0.001
    close_open_abs = np.abs(data['close'] - data['open'])
    
    # Calculate rolling max and min for 3-day window
    data['high_3d'] = data['high'].rolling(window=4, min_periods=1).max()
    data['low_3d'] = data['low'].rolling(window=4, min_periods=1).min()
    close_diff_3d = np.abs(data['close'] - data['close'].shift(3))
    range_3d = data['high_3d'] - data['low_3d'] + 0.001
    
    data['Multi_Timeframe_Volatility_Efficiency'] = (close_open_abs / high_low_range) * (close_diff_3d / range_3d)
    
    # Volatility_Momentum_Differential
    data['Volatility_Momentum_Differential'] = (
        (data['open'] - data['low']) * (data['close'] - data['open']) / high_low_range - 
        (data['high'] - data['close']) * (data['close'] - data['low']) / high_low_range
    )
    
    # Multi-Scale Volatility Momentum
    # Micro_Volatility_Momentum
    close_diff_1 = data['close'] - data['close'].shift(1)
    close_diff_2 = data['close'] - data['close'].shift(2)
    data['Micro_Volatility_Momentum'] = (close_diff_1 / high_low_range) * (close_diff_2 / (data['high'].shift(2) - data['low'].shift(2) + 0.001))
    
    # Meso_Volatility_Momentum
    data['high_8d'] = data['high'].rolling(window=9, min_periods=1).max()
    data['low_8d'] = data['low'].rolling(window=9, min_periods=1).min()
    data['high_13d'] = data['high'].rolling(window=14, min_periods=1).max()
    data['low_13d'] = data['low'].rolling(window=14, min_periods=1).min()
    
    close_diff_8 = data['close'] - data['close'].shift(8)
    close_diff_13 = data['close'] - data['close'].shift(13)
    data['Meso_Volatility_Momentum'] = (close_diff_8 / (data['high_8d'] - data['low_8d'] + 0.001)) * (close_diff_13 / (data['high_13d'] - data['low_13d'] + 0.001))
    
    # Macro_Volatility_Momentum
    data['high_21d'] = data['high'].rolling(window=22, min_periods=1).max()
    data['low_21d'] = data['low'].rolling(window=22, min_periods=1).min()
    data['high_34d'] = data['high'].rolling(window=35, min_periods=1).max()
    data['low_34d'] = data['low'].rolling(window=35, min_periods=1).min()
    
    close_diff_21 = data['close'] - data['close'].shift(21)
    close_diff_34 = data['close'] - data['close'].shift(34)
    data['Macro_Volatility_Momentum'] = (close_diff_21 / (data['high_21d'] - data['low_21d'] + 0.001)) * (close_diff_34 / (data['high_34d'] - data['low_34d'] + 0.001))
    
    # Volume Divergence Dynamics
    # Volume_Flow_Divergence
    close_diff_13_vol = data['close'] - data['close'].shift(13)
    data['Volume_Flow_Divergence'] = (
        data['volume'] * (data['close'] - data['open']) / high_low_range * 
        data['volume'] / (data['volume'].shift(13) + 0.001) * 
        close_diff_13_vol / (data['high_13d'] - data['low_13d'] + 0.001)
    )
    
    # Volume_Volatility_Efficiency
    vol_eff_t = data['volume'] / (high_low_range + 0.001)
    vol_eff_t1 = data['volume'].shift(1) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    data['Volume_Volatility_Efficiency'] = vol_eff_t / (vol_eff_t1 + 0.001)
    
    # Regime Transition Detection
    # Volatility_Regime_Persistence
    vol_range = data['high'] - data['low']
    vol_increase_count = pd.Series([0] * len(data), index=data.index)
    vol_decrease_count = pd.Series([0] * len(data), index=data.index)
    
    for i in range(5, len(data)):
        window = vol_range.iloc[i-5:i]
        vol_increase_count.iloc[i] = (window > window.shift(1)).sum()
        vol_decrease_count.iloc[i] = (window < window.shift(1)).sum()
    
    data['Volatility_Regime_Persistence'] = vol_increase_count - vol_decrease_count
    
    # Volume_Flow_Persistence
    vol_increase_count_8 = pd.Series([0] * len(data), index=data.index)
    vol_decrease_count_8 = pd.Series([0] * len(data), index=data.index)
    
    for i in range(8, len(data)):
        window = data['volume'].iloc[i-8:i]
        vol_increase_count_8.iloc[i] = (window > window.shift(1)).sum()
        vol_decrease_count_8.iloc[i] = (window < window.shift(1)).sum()
    
    data['Volume_Flow_Persistence'] = vol_increase_count_8 - vol_decrease_count_8
    
    # Price_Volume_Divergence_Strength
    close_diff_prev = data['close'] - data['close'].shift(1)
    vol_diff_prev = data['volume'] - data['volume'].shift(1)
    
    condition1 = (close_diff_prev > 0) & (vol_diff_prev < 0)
    condition2 = (close_diff_prev < 0) & (vol_diff_prev > 0)
    
    divergence_signal = np.where(condition1, 1, np.where(condition2, -1, 0))
    
    data['Price_Volume_Divergence_Strength'] = (
        divergence_signal * 
        np.abs(close_diff_prev) / high_low_range * 
        np.abs(vol_diff_prev) / (data['volume'].shift(1) + 0.001)
    )
    
    # Multi-Timeframe Integration
    data['Short_Term_Momentum'] = data['Volume_Flow_Divergence'] * data['Intraday_Volatility_Asymmetry'] * data['Micro_Volatility_Momentum']
    data['Medium_Term_Transition'] = data['Volume_Flow_Persistence'] * data['Multi_Timeframe_Volatility_Efficiency'] * data['Price_Volume_Divergence_Strength']
    data['Long_Term_Momentum'] = data['Volatility_Momentum_Differential'] * data['Macro_Volatility_Momentum'] * data['Volume_Volatility_Efficiency']
    
    # Adaptive Volatility Regime Alpha
    data['Expanding_Volatility_Regime'] = data['Short_Term_Momentum'] * data['Volatility_Regime_Persistence']
    data['Contracting_Volatility_Regime'] = data['Medium_Term_Transition'] * data['Long_Term_Momentum']
    
    # Final alpha calculation
    expanding_condition = data['Volatility_Regime_Persistence'] > 0
    contracting_condition = data['Volatility_Regime_Persistence'] < 0
    neutral_condition = data['Volatility_Regime_Persistence'] == 0
    
    data['Volatility_Regime_Momentum_Divime_Alpha'] = (
        expanding_condition * data['Expanding_Volatility_Regime'] +
        contracting_condition * data['Contracting_Volatility_Regime'] +
        neutral_condition * ((data['Expanding_Volatility_Regime'] + data['Contracting_Volatility_Regime']) / 2)
    )
    
    return data['Volatility_Regime_Momentum_Divime_Alpha']
