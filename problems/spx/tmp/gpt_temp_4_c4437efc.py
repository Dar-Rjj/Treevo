import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, lookback_period=20, amount_lookback_period_1=10, amount_lookback_period_2=30, volume_ma_window=10, price_vol_ratio_lookback=10):
    # Calculate daily return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Identify trend following momentum
    df['momentum'] = (df['close'] - df['close'].shift(lookback_period)) / df['close'].shift(lookback_period)
    
    # Measure intraday movement
    df['intraday_movement'] = df['high'] - df['low']
    
    # Analyze price volatility
    df['price_volatility'] = df['close'].pct_change().rolling(window=lookback_period).std()
    
    # Calculate daily return weighted by volume
    df['volume_weighted_return'] = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    
    # Compute volume change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Determine high volume days relative to average
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_window).mean()
    df['high_volume_day'] = (df['volume'] > df['volume_ma']).astype(int)
    
    # Analyze volume-weighted price
    df['volume_weighted_price'] = (df['close'] * df['volume']) / df['volume'].rolling(window=lookback_period).sum()
    
    # Find trade amount per unit of price
    df['amount_per_price_unit'] = df['amount'] / df['close']
    
    # Analyze trade amount trends
    df['amount_trend_1'] = (df['amount'] - df['amount'].shift(amount_lookback_period_1)) / df['amount'].shift(amount_lookback_period_1)
    df['amount_trend_2'] = (df['amount'] - df['amount'].shift(amount_lookback_period_2)) / df['amount'].shift(amount_lookback_period_2)
    
    # Combine with price and volume for more complex signals
    df['amount_volume_signal'] = (df['amount'] / df['close']) * (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Analyze the ratio of trade amount to volume
    df['amount_to_volume_ratio'] = df['amount'] / df['volume']
    
    # Evaluate the correlation between price changes and volume
    df['price_change'] = df['close'] - df['close'].shift(price_vol_ratio_lookback)
    df['volume_change_lagged'] = df['volume'] - df['volume'].shift(price_vol_ratio_lookback)
    df['price_volume_corr'] = df[['price_change', 'volume_change_lagged']].rolling(window=lookback_period).corr().iloc[::2, 1]
    
    # Investigate the interaction between intraday range and closing price
    df['intraday_range_to_close'] = (df['high'] - df['low']) / df['close']
    
    # Introduce a new factor: Price-Volume Ratio
    df['price_volume_ratio'] = (df['close'] / df['close'].shift(1)) * (df['volume'] / df['volume'].shift(1))
    
    # Final alpha factor
    df['alpha_factor'] = (df['daily_return'] + df['momentum'] + df['volume_weighted_return'] + 
                          df['volume_change'] + df['amount_trend_1'] + df['amount_trend_2'] + 
                          df['amount_volume_signal'] + df['amount_to_volume_ratio'] + 
                          df['price_volume_corr'] + df['intraday_range_to_close'] + 
                          df['price_volume_ratio']).fillna(0)
    
    return df['alpha_factor']
