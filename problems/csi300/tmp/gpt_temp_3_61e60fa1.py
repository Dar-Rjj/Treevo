import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum-Based Factors
    df['ROC_5'] = df['close'].pct_change(5)
    df['ROC_10'] = df['close'].pct_change(10)
    
    # Volatility Indicators
    df['Daily_Price_Range'] = df['high'] - df['low']
    
    # Volume Trend Signals
    df['PVI'] = 1000  # Initialize PVI
    df['NVI'] = 1000  # Initialize NVI
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'volume'] > df.loc[df.index[i-1], 'volume']:
            df.loc[df.index[i], 'PVI'] = df.loc[df.index[i-1], 'PVI'] * (df.loc[df.index[i], 'close'] / df.loc[df.index[i-1], 'close'])
            df.loc[df.index[i], 'NVI'] = df.loc[df.index[i-1], 'NVI']
        else:
            df.loc[df.index[i], 'NVI'] = df.loc[df.index[i-1], 'NVI'] * (df.loc[df.index[i], 'close'] / df.loc[df.index[i-1], 'close'])
            df.loc[df.index[i], 'PVI'] = df.loc[df.index[i-1], 'PVI']
    
    # Pattern Recognition Factors
    df['Bullish_Engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    df['Bearish_Engulfing'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    
    # Relative Strength Indicators
    # Assuming a benchmark asset close price is provided in the DataFrame
    if 'benchmark_close' in df.columns:
        df['Relative_Strength'] = (df['close'] / df['benchmark_close']) * 100
    else:
        df['Relative_Strength'] = 0  # Placeholder if benchmark_close is not available
    
    # Market Sentiment Measures
    df['OBV'] = 0  # Initialize OBV
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'close'] > df.loc[df.index[i-1], 'close']:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] + df.loc[df.index[i], 'volume']
        elif df.loc[df.index[i], 'close'] < df.loc[df.index[i-1], 'close']:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] - df.loc[df.index[i], 'volume']
        else:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV']
    
    # Volume-Weighted Indicators
    df['Cumulative_Volume_Delta'] = ((df['close'] > df['open']) * df['volume'] - (df['close'] < df['open']) * df['volume']).cumsum()
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Combine factors into a single alpha factor
    alpha_factor = (df['ROC_5'] + df['ROC_10'] + df['Daily_Price_Range'] + 
                    df['PVI'] + df['NVI'] + df['Bullish_Engulfing'] - df['Bearish_Engulfing'] + 
                    df['Relative_Strength'] + df['OBV'] + df['Cumulative_Volume_Delta'] + df['VWAP'])
    
    return alpha_factor
