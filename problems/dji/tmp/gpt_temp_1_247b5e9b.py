import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute the daily return
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    
    # Sum the daily returns over a 10-day window to capture medium-term momentum
    df['momentum_10d'] = df['daily_return'].rolling(window=10).sum()
    
    # Determine the True Range
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['prev_close']), abs(x[1] - df['prev_close'])), axis=1)
    
    # Calculate an Average True Range over a 21-day period
    df['atr_21d'] = df['true_range'].rolling(window=21).mean()
    
    # Identify days with unusually high volume compared to a 30-day average
    df['avg_vol_30d'] = df['volume'].rolling(window=30).mean()
    df['high_volume_day'] = df['volume'] > df['avg_vol_30d']
    
    # Examine if high volume days coincide with positive or negative price changes
    df['price_change'] = df['close'] - df['open']
    df['high_volume_signal'] = 0
    df.loc[df['high_volume_day'] & (df['price_change'] > 0), 'high_volume_signal'] = 1
    df.loc[df['high_volume_day'] & (df['price_change'] < 0), 'high_volume_signal'] = -1
    
    # Detect upward and downward gaps
    df['gap_up'] = (df['open'] > df['prev_close']) * (df['open'] - df['prev_close']) / df['prev_close']
    df['gap_down'] = (df['open'] < df['prev_close']) * (df['prev_close'] - df['open']) / df['prev_close']
    
    # Calculate the rate of change between current close and previous close
    df['roc'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # Categorize the rate of change into bins
    bins = [-float('inf'), -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, float('inf')]
    labels = [-4, -3, -2, -1, 0, 1, 2, 3]
    df['roc_score'] = pd.cut(df['roc'], bins=bins, labels=labels)
    
    # Measure the correlation coefficient between daily volume and close prices over a 30-day window
    df['vol_price_corr'] = df['volume'].rolling(window=30).corr(df['close'])
    
    # Calculate the VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Compare the current close price to the VWAP to determine overvalued or undervalued
    df['vwap_score'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        df['momentum_10d'] +
        df['high_volume_signal'] * 2 +
        df['gap_up'] * 3 -
        df['gap_down'] * 3 +
        df['roc_score'] * 2 +
        df['vol_price_corr'] +
        df['vwap_score']
    )
    
    return alpha_factor
