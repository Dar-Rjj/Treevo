import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Combined Price and Volume Momentum
    df['high_low_delta'] = df['high'] - df['low']
    df['close_open_delta'] = df['close'] - df['open']
    df['price_momentum_score'] = df['high_low_delta'] + df['close_open_delta']
    
    # Weighted by Previous Day's Momentum Score with Exponential Decay
    df['prev_momentum_score'] = df['price_momentum_score'].shift(1)
    decay_factor = 0.9
    df['weighted_momentum_score'] = df['price_momentum_score'] * (1 - decay_factor) + df['prev_momentum_score'] * decay_factor
    
    # Confirm with Volume Surge
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_weighted_momentum'] = df['weighted_momentum_score'] * df['volume_change']
    min_volume_increase = df['volume'].quantile(0.75)
    min_momentum_score = df['weighted_momentum_score'].quantile(0.75)
    df['confirmed_momentum'] = df[(df['volume_change'] > min_volume_increase) & (df['weighted_momentum_score'] > min_momentum_score)]['volume_weighted_momentum']
    
    # Dynamic Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['5_day_sum_price_change'] = df['price_change'].rolling(window=5).sum()
    df['10_day_ma'] = df['close'].rolling(window=10).mean()
    df['dynamic_momentum_score'] = df['5_day_sum_price_change'] - df['10_day_ma']
    
    # Adjust for VWAP
    df['vwap'] = ((df[['high', 'low', 'close']].mean(axis=1) * df['volume']).cumsum()) / df['volume'].cumsum()
    df['vwap_adjusted_momentum'] = df['dynamic_momentum_score'] / df['vwap']
    
    # Intraday Momentum and Volume Reversal
    df['high_low_range'] = df['high'] - df['low']
    df['high_close_ratio'] = df['high'] / df['close']
    df['volume_difference'] = df['volume'] - df['volume'].shift(1)
    df['volume_reversal_factor'] = np.sign(df['volume_difference'])
    df['intraday_factor'] = df['high_low_range'] * df['high_close_ratio'] * df['volume_reversal_factor']
    
    # Accumulated Momentum over Period
    n_days = 5
    df['accumulated_momentum'] = df['confirmed_momentum'].rolling(window=n_days).sum()
    df['max_volume_period'] = df['volume'].rolling(window=n_days).max()
    df['final_momentum_alpha'] = df['accumulated_momentum'] / df['max_volume_period']
    
    # Final Alpha Combination
    df['final_alpha'] = df['final_momentum_alpha'] * df['intraday_factor'] * df['vwap_adjusted_momentum']
    
    return df['final_alpha'].dropna()
