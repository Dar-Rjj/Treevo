import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Intraday Volatility Regime
    data['Range_Volatility'] = data['high'] - data['low']
    data['Gap_Volatility'] = abs(data['open'] - data['close'].shift(1))
    data['Volatility_Regime_Score'] = data['Range_Volatility'] / (data['Gap_Volatility'] + 0.001)
    
    # Multi-scale Volatility Persistence
    data['Range_Vol_SMA_3'] = data['Range_Volatility'].rolling(window=3, min_periods=1).mean()
    data['Range_Vol_SMA_5'] = data['Range_Volatility'].rolling(window=5, min_periods=1).mean()
    
    # Calculate persistence counts using rolling apply
    def count_above_3(x):
        return sum(x > data.loc[x.index, 'Range_Vol_SMA_3'])
    
    def count_above_5(x):
        return sum(x > data.loc[x.index, 'Range_Vol_SMA_5'])
    
    data['Short_Term_Vol_Persistence'] = data['Range_Volatility'].rolling(window=3, min_periods=1).apply(
        lambda x: sum(x > data.loc[x.index, 'Range_Vol_SMA_3']), raw=False
    )
    data['Medium_Term_Vol_Persistence'] = data['Range_Volatility'].rolling(window=5, min_periods=1).apply(
        lambda x: sum(x > data.loc[x.index, 'Range_Vol_SMA_5']), raw=False
    )
    data['Vol_Persistence_Ratio'] = data['Short_Term_Vol_Persistence'] / (data['Medium_Term_Vol_Persistence'] + 0.001)
    
    # Price-Volume Divergence Dynamics
    # Directional Volume Flow
    price_diff = data['close'] - data['open']
    data['Up_Move_Volume'] = price_diff * data['volume'] / (abs(price_diff) + 0.001)
    data['Down_Move_Volume'] = (-price_diff) * data['volume'] / (abs(price_diff) + 0.001)
    data['Volume_Direction_Ratio'] = data['Up_Move_Volume'] / (data['Down_Move_Volume'] + 0.001)
    
    # Price-Volume Efficiency Divergence
    data['Price_Efficiency'] = price_diff / (data['Range_Volatility'] + 0.001)
    data['Volume_SMA_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['Volume_Efficiency'] = data['volume'] / (data['Volume_SMA_5'] + 0.001)
    data['Efficiency_Divergence'] = data['Price_Efficiency'] * data['Volume_Efficiency']
    
    # Regime-Adaptive Signal Generation
    # High Volatility Signals
    data['Vol_Spike_Indicator'] = (data['Range_Volatility'] > data['Range_Vol_SMA_5']) * data['Volume_Efficiency']
    data['Gap_Recovery_Signal'] = ((data['close'] > data['open']) & 
                                 (data['open'] < data['close'].shift(1))) * data['Volume_Direction_Ratio']
    data['High_Vol_Signal'] = data['Vol_Spike_Indicator'] * data['Gap_Recovery_Signal']
    
    # Low Volatility Signals
    data['Range_Compression_Score'] = (data['Range_Volatility'] < data['Range_Vol_SMA_5']) * data['Vol_Persistence_Ratio']
    data['Breakout_Potential'] = price_diff * data['Volume_Direction_Ratio']
    data['Low_Vol_Signal'] = data['Range_Compression_Score'] * data['Breakout_Potential']
    
    # Multi-timeframe Confirmation
    # Short-term Confirmation
    data['Price_Confirmation_Short'] = ((data['close'] > data['close'].shift(1)) & 
                                      (data['close'].shift(1) > data['close'].shift(2))).astype(float)
    data['Volume_Confirmation_Short'] = ((data['volume'] > data['volume'].shift(1)) & 
                                       (data['volume'].shift(1) > data['volume'].shift(2))).astype(float)
    data['Short_Term_Confirm'] = data['Price_Confirmation_Short'] * data['Volume_Confirmation_Short']
    
    # Medium-term Alignment
    data['Close_SMA_5'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['Price_Trend_Medium'] = (data['close'] > data['Close_SMA_5']).astype(float) - (data['close'] < data['Close_SMA_5']).astype(float)
    data['Volume_Trend_Medium'] = (data['volume'] > data['Volume_SMA_5']).astype(float) - (data['volume'] < data['Volume_SMA_5']).astype(float)
    data['Medium_Term_Align'] = data['Price_Trend_Medium'] * data['Volume_Trend_Medium']
    
    # Adaptive Alpha Construction
    data['Volatility_Regime_Component'] = data['Volatility_Regime_Score'] * data['Vol_Persistence_Ratio']
    data['Divergence_Component'] = data['Volume_Direction_Ratio'] * data['Efficiency_Divergence']
    data['Signal_Component'] = data['High_Vol_Signal'] + data['Low_Vol_Signal']
    data['Confirmation_Component'] = data['Short_Term_Confirm'] * data['Medium_Term_Align']
    
    # Final Alpha Factor
    alpha = (data['Volatility_Regime_Component'] * 
             data['Divergence_Component'] * 
             data['Signal_Component'] * 
             data['Confirmation_Component'])
    
    return alpha
