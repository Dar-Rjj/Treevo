import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Reversal Signal
    df['reversal_signal'] = df['intraday_return'].apply(lambda x: -1 if x > 0 else 1)
    
    # Calculate Daily Price Return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate 20-Day Weighted Moving Average of Returns
    df['weighted_return'] = (df['daily_return'] * df['volume']).rolling(window=22).sum() / df['volume'].rolling(window=22).sum()
    
    # Adjust for Price Volatility
    df['price_range'] = df['high'] - df['low']
    df['avg_price_range'] = df['price_range'].rolling(window=22).mean()
    df['final_factor_value'] = df['weighted_return'] - df['avg_price_range']
    
    # Introduce Volume-Weighted Price Change
    df['volume_weighted_price'] = (df['close'] * df['volume']).rolling(window=22).sum() / df['volume'].rolling(window=22).sum()
    df['volume_weighted_change'] = df['close'] - df['volume_weighted_price']
    
    # Combine Factors
    df['atr'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    df['atr_reversal'] = df['atr'] * df['reversal_signal']
    df['ema_atr_reversal'] = df['atr_reversal'].ewm(span=10, adjust=False).mean()
    
    # Simple Moving Averages
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Momentum and Volume Indicators
    df['momentum_vol_indicator'] = (
        0.3 * df['weighted_return'] + 
        0.2 * (df['sma_100'] - df['sma_200']) + 
        0.2 * (df['sma_50'] - df['sma_100']) + 
        df['ema_atr_reversal']
    )
    
    # Introduce Volume-Weighted Momentum
    df['volume_weighted_momentum'] = (df['daily_return'] * df['volume']).rolling(window=22).sum() / df['volume'].rolling(window=22).sum()
    df['final_factor_value'] += df['volume_weighted_momentum']
    
    # Final Factor
    df['alpha_factor'] = (
        df['reversal_signal'] + 
        df['weighted_return'] + 
        df['final_factor_value'] + 
        df['volume_weighted_change'] + 
        df['momentum_vol_indicator']
    )
    
    return df['alpha_factor']
