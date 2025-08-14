import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate trend following momentum with a lookback period of 20 days
    df['trend_momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Calculate intraday movement
    df['intraday_movement'] = df['high'] - df['low']
    
    # Calculate volume-weighted daily return
    df['volume_weighted_daily_return'] = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    
    # Calculate volume change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Identify high volume days using a 20-day moving average
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['high_volume_day'] = (df['volume'] > df['avg_volume_20']).astype(int)
    
    # Calculate trade amount per unit of price
    df['amount_per_price'] = df['amount'] / df['close']
    
    # Calculate amount trend with a lookback period of 20 days
    df['amount_trend'] = (df['amount'] - df['amount'].shift(20)) / df['amount'].shift(20)
    
    # Calculate the correlation between price change and volume change with a lag of 5 days
    df['price_change'] = df['close'] - df['close'].shift(5)
    df['volume_change_lag'] = df['volume'] - df['volume'].shift(5)
    df['price_vol_corr'] = df[['price_change', 'volume_change_lag']].rolling(window=20).corr().iloc[::2, :]['price_change']
    
    # Calculate intraday range relative to closing price
    df['intraday_range_rel_close'] = (df['high'] - df['low']) / df['close']
    
    # Calculate price-volume ratio
    df['price_volume_ratio'] = (df['close'] / df['close'].shift(1)) * (df['volume'] / df['volume'].shift(1))
    
    # Combine multiple factors into a single alpha factor
    df['alpha_factor'] = (
        df['daily_return'] +
        df['trend_momentum'] +
        df['intraday_movement'] +
        df['volume_weighted_daily_return'] +
        df['volume_change'] +
        df['high_volume_day'] +
        df['amount_per_price'] +
        df['amount_trend'] +
        df['price_vol_corr'] +
        df['intraday_range_rel_close'] +
        df['price_volume_ratio']
    ) / 11  # Normalize by the number of factors
    
    return df['alpha_factor']
