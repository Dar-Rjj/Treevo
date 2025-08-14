import numpy as np
def heuristics_v2(df):
    # Daily Volume Change
    df['daily_volume_change'] = df['volume'].diff()
    
    # 5-Day Volume Changes and 5-Day Log Return
    df['5_day_vol_change'] = df['daily_volume_change'].rolling(window=5).sum()
    df['5_day_log_return'] = np.log(df['close'] / df['close'].shift(5))
