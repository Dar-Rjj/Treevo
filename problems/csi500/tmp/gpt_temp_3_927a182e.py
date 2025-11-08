import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Components
    data['raw_intraday_return'] = data['close'] - data['open']
    data['scaled_intraday_return'] = np.where(
        (data['high'] - data['low']) > 0,
        (data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    data['close_acceleration'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    
    # Intraday Reversal Components
    data['price_range_momentum'] = (data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))
    data['intraday_reversal_signal'] = np.where(
        (data['high'] + data['low']) > 0,
        (data['high'] - data['low']) / (data['high'] + data['low']),
        0
    )
    data['momentum_reversal_divergence'] = data['scaled_intraday_return'] * data['intraday_reversal_signal']
    
    # Volume Confirmation System
    data['volume_efficiency'] = np.where(
        (data['high'] - data['low']) > 0,
        data['volume'] / (data['high'] - data['low']),
        0
    )
    data['volume_trend'] = data['volume'] / data['volume'].shift(1)
    data['volume_price_alignment'] = np.sign(data['scaled_intraday_return']) * data['volume_trend']
    data['volume_confirmation_score'] = data['volume_efficiency'] * data['volume_price_alignment']
    
    # Range Efficiency Analysis
    data['price_completion_ratio'] = np.where(
        (data['high'] - data['low']) > 0,
        (data['close'] - data['low']) / (data['high'] - data['low']),
        0.5
    )
    
    # Range Persistence calculation
    completion_threshold = 0.1
    data['completion_pattern'] = np.where(
        data['price_completion_ratio'] > 0.5 + completion_threshold, 1,
        np.where(data['price_completion_ratio'] < 0.5 - completion_threshold, -1, 0)
    )
    
    # Calculate consecutive similar patterns
    data['range_persistence'] = 0
    for i in range(2, len(data)):
        if data['completion_pattern'].iloc[i] == data['completion_pattern'].iloc[i-1]:
            data.loc[data.index[i], 'range_persistence'] = data['range_persistence'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'range_persistence'] = 1
    
    data['range_efficiency_score'] = data['price_completion_ratio'] * (1 + data['range_persistence'] / 10)
    
    # Liquidity Efficiency Components
    data['dollar_volume'] = data['close'] * data['volume']
    data['amount_efficiency'] = np.where(
        (data['high'] - data['low']) > 0,
        data['amount'] / (data['high'] - data['low']),
        0
    )
    data['combined_liquidity_measure'] = data['dollar_volume'] * data['amount_efficiency']
    
    # Signal Integration Logic
    data['core_momentum_reversal'] = (
        data['raw_intraday_return'] * 
        data['price_range_momentum'] * 
        data['momentum_reversal_divergence']
    )
    
    data['volume_weighted_enhancement'] = data['volume_confirmation_score'] * data['volume_trend']
    
    data['liquidity_adjustment'] = np.where(
        data['combined_liquidity_measure'] > 0,
        1 / data['combined_liquidity_measure'],
        1
    )
    
    # Final Alpha Generation
    data['combined_signal'] = (
        data['core_momentum_reversal'] * 
        data['volume_weighted_enhancement'] * 
        data['range_efficiency_score'] * 
        data['liquidity_adjustment']
    )
    
    # Volume threshold filter (only use signals when volume is above 20-day rolling average)
    volume_threshold = data['volume'].rolling(window=20, min_periods=1).mean()
    data['final_alpha'] = np.where(
        data['volume'] > volume_threshold,
        data['combined_signal'],
        0
    )
    
    return data['final_alpha']
