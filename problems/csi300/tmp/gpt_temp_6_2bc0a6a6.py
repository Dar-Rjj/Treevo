import pandas as pd

def heuristics_v2(df):
    # Define the heuristics matrix
    heuristics_matrix = pd.Series(index=df.index, dtype=float)
    
    # Example heuristic: difference between today's close and yesterday's close
    heuristics_matrix['close_diff'] = df['close'].diff()
    
    # Example heuristic: ratio of today's volume to the 5-day moving average volume
    heuristics_matrix['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Example heuristic: (High - Low) / Close for each day
    heuristics_matrix['hl_over_close'] = (df['high'] - df['low']) / df['close']
    
    # Example heuristic: (Close - Open) / Open for each day
    heuristics_matrix['co_over_open'] = (df['close'] - df['open']) / df['open']
    
    # Example heuristic: (Amount - Volume) / Volume for each day
    heuristics_matrix['av_over_volume'] = (df['amount'] - df['volume']) / df['volume']
    
    return heuristics_matrix
