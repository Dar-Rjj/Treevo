import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate weighted moving averages (WMA)
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    # 10-day WMA
    df['wma_10'] = wma(df['close'], 10)
    # 20-day WMA
    df['wma_20'] = wma(df['close'], 20)
    
    # Determine if 10-day WMA is above 20-day WMA (Bullish Signal)
    df['wma_signal'] = (df['wma_10'] > df['wma_20']).astype(int)
    
    # Calculate rate of change (ROC)
    def roc(data, period):
        return (data - data.shift(period)) / data.shift(period)
    
    # ROC over a 5-day period
    df['roc_5'] = roc(df['close'], 5)
    # ROC over a 14-day period
    df['roc_14'] = roc(df['close'], 14)
    
    # Calculate volume-weighted average price (VWAP)
    def vwap(data, window):
        return (data['volume'] * data['close']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    
    # VWAP over a 10-day window
    df['vwap_10'] = vwap(df, 10)
    # VWAP over a 20-day window
    df['vwap_20'] = vwap(df, 20)
    
    # Correlate volume-weighted average price with price changes
    df['vwap_corr'] = df[['vwap_10', 'vwap_20', 'close']].pct_change().rolling(window=10).corr().unstack().iloc[::3, 2].fillna(0)
    
    # Identify bullish hammer pattern
    df['hammer'] = ((df['close'] > df['open']) & 
                    (df['low'] < df['close'] - 2 * (df['close'] - df['open']))).astype(int)
    
    # Identify bearish hanging man pattern
    df['hanging_man'] = ((df['close'] < df['open']) & 
                         (df['low'] < df['close'] - 2 * (df['open'] - df['close']))).astype(int)
    
    # Calculate standard deviation of daily returns
    df['return'] = df['close'].pct_change()
    # Standard deviation over a 10-day window
    df['std_10'] = df['return'].rolling(window=10).std()
    # Standard deviation over a 20-day window
    df['std_20'] = df['return'].rolling(window=20).std()
    
    # Average high-low range over a 10-day window
    df['hl_range_10'] = (df['high'] - df['low']).rolling(window=10).mean()
    # Average high-low range over a 20-day window
    df['hl_range_20'] = (df['high'] - df['low']).rolling(window=20).mean()
    
    # Final alpha factor
    df['alpha_factor'] = (df['wma_signal'] + df['roc_5'] + df['roc_14'] + df['vwap_corr'] + 
                          df['hammer'] - df['hanging_man'] + df['std_10'] + df['hl_range_10'])
    
    return df['alpha_factor']
