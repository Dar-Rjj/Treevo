import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the rate of change (ROC) over different time periods
    df['ROC_10'] = df['close'].pct_change(10)
    df['ROC_20'] = df['close'].pct_change(20)
    df['ROC_50'] = df['close'].pct_change(50)
    
    # Price difference between close of day t and close of day t-10, t-20, t-50
    df['price_diff_10'] = df['close'] - df['close'].shift(10)
    df['price_diff_20'] = df['close'] - df['close'].shift(20)
    df['price_diff_50'] = df['close'] - df['close'].shift(50)
    
    # Volume-weighted average price over different time periods
    df['VWAP_10'] = (df['close'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    df['VWAP_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['VWAP_50'] = (df['close'] * df['volume']).rolling(window=50).sum() / df['volume'].rolling(window=50).sum()
    
    # ROC of volume-weighted average price
    df['ROC_VWAP_10'] = df['VWAP_10'].pct_change(10)
    df['ROC_VWAP_20'] = df['VWAP_20'].pct_change(20)
    df['ROC_VWAP_50'] = df['VWAP_50'].pct_change(50)
    
    # Range (high - low) and its rate of change
    df['range'] = df['high'] - df['low']
    df['ROC_range_10'] = df['range'].pct_change(10)
    df['ROC_range_20'] = df['range'].pct_change(20)
    df['ROC_range_50'] = df['range'].pct_change(50)
    
    # Difference between open and close prices
    df['open_close_diff'] = df['open'] - df['close']
    df['ROC_open_close_10'] = df['open_close_diff'].pct_change(10)
    df['ROC_open_close_20'] = df['open_close_diff'].pct_change(20)
    df['ROC_open_close_50'] = df['open_close_diff'].pct_change(50)
    
    # Rate of change in volume
    df['ROC_volume_10'] = df['volume'].pct_change(10)
    df['ROC_volume_20'] = df['volume'].pct_change(20)
    df['ROC_volume_50'] = df['volume'].pct_change(50)
    
    # Volume on days with positive and negative price change
    df['positive_vol'] = df.apply(lambda x: x['volume'] if x['close'] > x['close'].shift(1) else 0, axis=1)
    df['negative_vol'] = df.apply(lambda x: x['volume'] if x['close'] < x['close'].shift(1) else 0, axis=1)
    df['vol_ratio'] = df['positive_vol'].rolling(window=10).sum() / df['negative_vol'].rolling(window=10).sum()
    
    # Composite score by combining price momentum, volume, and range
    df['composite_score'] = (df['ROC_10'] + df['ROC_volume_10'] + df['ROC_range_10']) / 3
    df['composite_score_weighted'] = 0.5 * df['ROC_10'] + 0.3 * df['ROC_volume_10'] + 0.2 * df['ROC_range_10']
    
    return df['composite_score_weighted']
