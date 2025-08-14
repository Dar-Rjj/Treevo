import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback_period=20, amount_lookback_period=10, volume_window=20):
    # Calculate daily return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Identify trend following momentum
    df['momentum'] = (df['close'] - df['close'].shift(lookback_period)) / df['close'].shift(lookback_period)
    
    # Measure intraday movement
    df['intraday_movement'] = df['high'] - df['low']
    
    # Calculate daily return weighted by volume
    df['volume_weighted_return'] = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    
    # Compute volume change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Determine high volume days relative to average
    df['average_volume'] = df['volume'].rolling(window=volume_window).mean()
    df['high_volume_day'] = (df['volume'] > df['average_volume']).astype(int)
    
    # Find trade amount per unit of price
    df['trade_amount_per_price'] = df['amount'] / df['close']
    
    # Analyze trade amount trends
    df['amount_trend'] = (df['amount'] - df['amount'].shift(amount_lookback_period)) / df['amount'].shift(amount_lookback_period)
    
    # Combine with price and volume for more complex signals
    df['complex_signal'] = (df['amount'] / df['close']) * (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Evaluate the correlation between price changes and volume
    df['price_change'] = df['close'] - df['close'].shift(lookback_period)
    df['volume_change_lagged'] = df['volume'] - df['volume'].shift(lookback_period)
    df['price_volume_correlation'] = df[['price_change', 'volume_change_lagged']].corr().unstack().iloc[1::2].values
    
    # Investigate the interaction between intraday range and closing price
    df['intraday_range_to_close'] = (df['high'] - df['low']) / df['close']
    
    # Introduce a new factor: Price-Volume Ratio
    df['price_volume_ratio'] = (df['close'] / df['close'].shift(1)) * (df['volume'] / df['volume'].shift(1))
    
    # Construct a combined momentum-volatility index
    df['momentum_volatility_index'] = (df['close'] - df['close'].shift(lookback_period)) / df['close'].shift(lookback_period) + (df['high'] - df['low']) / df['close']
    
    # Create a high-volume adjusted return
    df['high_volume_adjusted_return'] = df['daily_return'].where(df['high_volume_day'] == 1, 0)
    
    # Formulate a comprehensive price-amount-volume indicator
    df['comprehensive_indicator'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * (df['amount'] / df['close']) * (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Final alpha factor
    df['alpha_factor'] = (
        df['momentum'] + 
        df['intraday_movement'] + 
        df['volume_weighted_return'] + 
        df['volume_change'] + 
        df['high_volume_day'] + 
        df['trade_amount_per_price'] + 
        df['amount_trend'] + 
        df['complex_signal'] + 
        df['price_volume_correlation'] + 
        df['intraday_range_to_close'] + 
        df['price_volume_ratio'] + 
        df['momentum_volatility_index'] + 
        df['high_volume_adjusted_return'] + 
        df['comprehensive_indicator']
    )
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
