import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['price_momentum'] = df['log_return'].rolling(window=5).sum()

    # Apply Combined Volume and Amount Shock Filter
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)).abs()
    df['amount_change'] = (df['amount'] - df['amount'].shift(1)).abs()
    
    volume_threshold = df['volume_change'].quantile(0.75)
    amount_threshold = df['amount_change'].quantile(0.75)
    
    df = df[(df['volume_change'] < volume_threshold) & (df['amount_change'] < amount_threshold)]

    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'] - df['open']
    
    # Calculate Volume-Weighted Moving Average of Intraday Volatility
    df['vol_weighted_intraday_volatility'] = df['intraday_volatility'] * df['volume']
    df['vol_weighted_intraday_volatility_ma'] = df['vol_weighted_intraday_volatility'].rolling(window=10).mean()
    
    # Calculate Intraday Reversal Score
    df['reversal_score'] = df['close_to_open_return'] / df['vol_weighted_intraday_volatility_ma']
    
    # Apply Exponential Decay to Reversal Score
    half_life = 5
    df['exponential_decay_factor'] = np.exp(-np.log(2) / half_life)
    df['decayed_reversal_score'] = df['reversal_score'].ewm(alpha=(1 - df['exponential_decay_factor']), adjust=False).mean()
    
    return df['decayed_reversal_score']
