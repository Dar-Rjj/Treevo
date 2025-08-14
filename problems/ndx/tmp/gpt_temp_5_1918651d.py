import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=14):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate Volume Change Ratio
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['weighted_momentum'] = (df['daily_return'] * df['volume_change_ratio']).rolling(window=n).sum()
    
    # Adjust for Price Volatility
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['atr'] = df['true_range'].rolling(window=m).mean()
    df['atr_adjusted_momentum'] = df['weighted_momentum'] - df['atr']
    
    # Calculate 5-Day Simple Moving Average (SMA) of Close price
    df['sma_5'] = df['close'].rolling(window=5).mean()
    
    # Compute Price Difference
    df['price_difference'] = df['close'] - df['sma_5']
    
    # Compute Momentum Score
    df['momentum_score'] = df['price_difference'] / df['sma_5']
    
    # Calculate Cumulative Volume
    df['cumulative_volume'] = df['volume'].rolling(window=5).sum()
    
    # Adjust Momentum Score by Cumulative Volume
    df['adjusted_momentum_score'] = df['momentum_score'] * df['cumulative_volume']
    
    # Calculate High-to-Low Price Range
    df['range'] = df['high'] - df['low']
    
    # Calculate Trading Intensity
    df['volume_change'] = df['volume'].diff()
    df['amount_change'] = df['amount'].diff()
    df['trading_intensity'] = (df['volume_change'] / df['amount_change']).fillna(0)
    
    # Weight the Range by Trading Intensity
    df['weighted_range'] = df['range'] * (df['trading_intensity'] * 1000)
    
    # Combine Adjusted Momentum, Weighted Range, and ATR-Adjusted Momentum
    df['alpha_factor'] = df['adjusted_momentum_score'] + df['weighted_range'] + df['atr_adjusted_momentum']
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
