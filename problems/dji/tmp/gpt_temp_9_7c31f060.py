import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-PreviousClose Return
    df['close_to_previous_close_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Combine Intraday and Close-to-PreviousClose Returns
    df['combined_return'] = df['intraday_return'] + df['close_to_previous_close_return']
    
    # Weight by Volume
    df['volume_weighted_return'] = df['combined_return'] * df['volume']
    
    # Calculate Daily Trading Range Adjusted by Volume
    df['trading_range'] = df['high'] - df['low']
    df['average_volume'] = df['volume'].rolling(window=20).mean()
    df['trading_range_adjusted_by_volume'] = df['trading_range'] * (df['volume'] / df['average_volume'])
    
    # Adjust for Volatility
    df['daily_range_volatility'] = df['trading_range'].rolling(window=20).std()
    df['volatility_adjusted_return'] = df['volume_weighted_return'] / df['daily_range_volatility']
    
    # Calculate Price Volatility Indicator (PVI)
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['pvi'] = df['tr'] / df['atr']
    df['pvi_smoothed'] = df['pvi'].ewm(span=7, adjust=False).mean()
    
    # Combine Volume-Weighted Returns and PVI
    df['factor'] = df['volatility_adjusted_return'] * df['pvi_smoothed']
    df['final_factor'] = df['factor'] - df['factor'].rolling(window=30).mean()
    
    return df['final_factor']
