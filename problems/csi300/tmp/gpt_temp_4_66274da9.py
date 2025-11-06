import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Liquidity Absorption Analysis
    data['Price_Absorption_Ratio'] = (data['close'] - data['open']) * data['volume'] / (data['amount'] + 0.0001)
    data['High_Low_Absorption'] = (data['high'] - data['low']) * data['volume'] / (data['amount'] + 0.0001)
    absorption_diff = data['Price_Absorption_Ratio'] - data['High_Low_Absorption']
    data['Absorption_Divergence'] = np.sign(absorption_diff) * np.log(1 + np.abs(absorption_diff))
    
    # Momentum Persistence Framework
    data['Short_term_Momentum'] = (data['close'] - data['close'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2) + 0.0001)
    
    # Calculate rolling high-low range for medium-term momentum
    high_low_range = data['high'] - data['low']
    data['Medium_term_Momentum'] = (data['close'] - data['close'].shift(5)) / (high_low_range.rolling(window=6, min_periods=1).sum().shift(1) + 0.0001)
    
    # Momentum Consistency
    momentum_product = data['Short_term_Momentum'] * data['Medium_term_Momentum']
    data['Momentum_Consistency'] = np.sign(momentum_product) * np.minimum(np.abs(data['Short_term_Momentum']), np.abs(data['Medium_term_Momentum']))
    
    # Volume-Price Efficiency System
    data['Volume_Efficiency'] = np.abs(data['close'] - data['close'].shift(1)) * data['volume'] / (data['amount'] + 0.0001)
    data['Gap_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001)
    efficiency_diff = np.abs(data['Volume_Efficiency'] - data['Gap_Efficiency'])
    data['Efficiency_Convergence'] = (data['Volume_Efficiency'] * data['Gap_Efficiency']) / (1 + efficiency_diff)
    
    # Market Microstructure Patterns
    data['Opening_Gap_Persistence'] = np.sign(data['close'] - data['open']) * np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001)
    data['Intraday_Reversal_Strength'] = (np.maximum(data['open'], data['close']) - np.minimum(data['open'], data['close'])) / (data['high'] - data['low'] + 0.0001)
    data['Microstructure_Momentum'] = data['Opening_Gap_Persistence'] * (1 - data['Intraday_Reversal_Strength']) * data['volume'] / (data['volume'].shift(1) + 0.0001)
    
    # Volatility-Liquidity Interaction
    # Volatility Compression
    rolling_high_low = (data['high'] - data['low']).rolling(window=4, min_periods=1)
    data['Volatility_Compression'] = (data['high'] - data['low']) / (rolling_high_low.apply(lambda x: x.iloc[:-1].max() if len(x) > 1 else 1.0) + 0.0001)
    
    # Liquidity Expansion
    rolling_volume = data['volume'].rolling(window=4, min_periods=1)
    data['Liquidity_Expansion'] = data['volume'] / (rolling_volume.apply(lambda x: x.iloc[:-1].max() if len(x) > 1 else 1.0) + 0.0001)
    
    data['Compression_Expansion_Ratio'] = data['Volatility_Compression'] / (data['Liquidity_Expansion'] + 0.0001)
    
    # Alpha Integration
    # Primary Signals
    primary_signal_1 = data['Absorption_Divergence'] * data['Momentum_Consistency']
    primary_signal_2 = data['Efficiency_Convergence'] * data['Microstructure_Momentum']
    
    # Market Condition Modifier
    market_modifier = 1.0 + 0.5 * np.tanh(data['Compression_Expansion_Ratio'] - 1) + 0.3 * (1 - np.exp(-np.abs(data['Momentum_Consistency'])))
    
    # Final Alpha
    final_alpha = (primary_signal_1 + primary_signal_2) * market_modifier * np.sign(data['Absorption_Divergence']) * (1 + np.log(1 + np.abs(data['Compression_Expansion_Ratio'])))
    
    return final_alpha
