import pandas as pd
import pandas as pd

def heuristics(df, n=5):
    # Calculate Volume-Weighted Average Prices
    df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume_weighted_price'] = df['avg_price'] * df['volume']
    
    # Sum Over Last N Days
    sum_volume_weighted_price = df['volume_weighted_price'].rolling(window=n).sum()
    sum_volume = df['volume'].rolling(window=n).sum()
    
    # Volume-Weighted Moving Average
    vwma = sum_volume_weighted_price / sum_volume
    
    # Current Day's Volume-Weighted Price
    current_vw_price = df['avg_price'] * df['volume']
    
    # VWPTI
    vwpti = (current_vw_price - vwma) / vwma
    
    # Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Adjust for Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['open']
    adjusted_intraday_return = intraday_return / intraday_volatility
    
    # Volume Trend Factor
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    volume_trend_factor = (df['volume'] - avg_volume_5d) * adjusted_intraday_return
    
    # Amount Momentum Factor
    daily_amount = (df['high'] + df['low']) / 2 * df['volume']
    avg_amount_5d = daily_amount.rolling(window=5).mean()
    amount_momentum_factor = (daily_amount - avg_amount_5d) * adjusted_intraday_return
    
    # Identify Directional Days
    df['direction'] = df.apply(lambda row: 'Up' if row['close'] > row['open'] else 'Down', axis=1)
    up_count = df['direction'].rolling(window=n).apply(lambda x: (x == 'Up').sum(), raw=False)
    down_count = df['direction'].rolling(window=n).apply(lambda x: (x == 'Down').sum(), raw=False)
    directional_count = (up_count - down_count) * df['volume']
    
    # Combine Components
    factor = vwpti + volume_trend_factor + amount_momentum_factor + directional_count
    
    return factor

# Example usage:
# df = pd.read_csv('stock_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics(df)
# print(factor_values)
