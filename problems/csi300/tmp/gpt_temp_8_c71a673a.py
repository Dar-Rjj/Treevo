import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Adjust Window Size Based on Volatility
    def get_window_size(volatility):
        if volatility > df['volatility'].median():
            return 10
        else:
            return 30
    
    # Market Regime Detection
    df['market_regime'] = np.nan
    df.loc[df['close'] > df[['high', 'low', 'close']].mean(axis=1), 'market_regime'] = 'Bullish'
    df.loc[df['close'] < df[['high', 'low', 'close']].mean(axis=1), 'market_regime'] = 'Bearish'
    df['market_regime'] = df['market_regime'].fillna('Sideways')
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = np.nan
    df['rolling_std'] = np.nan
    for i in range(30, len(df)):
        window = get_window_size(df.loc[df.index[i], 'volatility'])
        df.loc[df.index[i], 'rolling_mean'] = df.loc[df.index[i-window]:df.index[i], 'volume_weighted_return'].mean()
        df.loc[df.index[i], 'rolling_std'] = df.loc[df.index[i-window]:df.index[i], 'volume_weighted_return'].std()
    
    # Final Alpha Factor
    df['alpha_factor'] = df['rolling_mean'] / df['rolling_std']
    
    return df['alpha_factor'].dropna()
