import pandas as pd
import pandas as pd

def heuristics_v2(df, N=50, M=10, K=14, L=14, P=20, Q=5, R=5):
    # Momentum-based factors
    df['SMA'] = df['close'].rolling(window=N).mean()
    df['SMA_diff'] = df['close'] - df['SMA']
    df['ROC'] = df['close'].pct_change(periods=M)
    
    # Volatility-based factors
    true_range = pd.concat([df['high']-df['low'], 
                            abs(df['high']-df['close'].shift(1)), 
                            abs(df['low']-df['close'].shift(1))], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=L).mean()
    
    # Volume-based factors
    df['vol_ma'] = df['volume'].rolling(window=P).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    df['up_vol'] = df.apply(lambda x: x['volume'] if x['close'] > x['open'] else 0, axis=1)
    df['down_vol'] = df.apply(lambda x: x['volume'] if x['close'] < x['open'] else 0, axis=1)
    df['cum_up_vol'] = df['up_vol'].rolling(window=Q).sum()
    df['cum_down_vol'] = df['down_vol'].rolling(window=Q).sum()
    
    # Pattern recognition for alpha generation
    def is_doji(row, threshold=0.01):
        return (row['open'] - row['close']) / ((row['high'] - row['low']) or 1) < threshold
    
    def is_hammer(row, body_threshold=0.3, lower_shadow_threshold=2):
        return (row['close'] - row['open']) / (row['high'] - row['low']) > body_threshold and \
               (row['open'] - row['low']) / (row['close'] - row['high']) > lower_shadow_threshold
    
    def is_engulfing(row, prev_row):
        return (row['close'] > row['open'] and prev_row['close'] < prev_row['open'] and
                row['close'] > prev_row['open'] and row['open'] < prev_row['close'])
    
    df['doji'] = df.apply(is_doji, axis=1).astype(int)
    df['hammer'] = df.apply(is_hammer, axis=1).astype(int)
    df['engulfing'] = df.apply(lambda x: is_engulfing(x, df.loc[x.name - pd.DateOffset(1)]) if x.name in df.index else 0, axis=1).fillna(0).astype(int)
    
    df['pattern_score'] = df['doji'] + df['hammer'] + df['engulfing']
    df['bullish_signals'] = df['hammer'] + df['engulfing'].where(df['close'] > df['open'], 0)
    df['bearish_signals'] = df['engulfing'].where(df['close'] < df['open'], 0)
    
    # Generate the final factor
    df['factor'] = (df['SMA_diff'] + df['ROC'] + df['ATR'] + df['vol_ratio'] +
                    df['cum_up_vol'] - df['cum_down_vol'] + df['pattern_score'] + 
                    df['bullish_signals'] - df['bearish_signals'])
    
    return df['factor']
