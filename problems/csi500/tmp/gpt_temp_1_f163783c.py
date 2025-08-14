import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 30-day High-Low Range Momentum
    df['range_momentum'] = df['high_low_range'].rolling(window=30).apply(lambda x: x[-1] - x[0], raw=True)
    
    # Calculate 30-day Close Price Momentum
    df['close_momentum'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Combine High-Low Range and Close Price Momentum
    df['integrated_momentum'] = (df['range_momentum'] + df['close_momentum']) / 2
    
    # Identify Volume Shock
    df['avg_volume_30'] = df['volume'].rolling(window=30, min_periods=1).mean()
    df['volume_shock'] = (df['volume'] > 2 * df['avg_volume_30']).astype(int)
    
    # Integrate Momentum and Volume Shock
    df['integrated_momentum_vol_shock'] = df['integrated_momentum'] * df['volume_shock']
    
    # Calculate Intraday Return
    df['intraday_return'] = df['high'] - df['low']
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / (df['close'] + df['open'])
    
    # Adjust Intraday Return by Intraday Volatility
    df['intraday_return_adjusted'] = df['intraday_return'] / df['intraday_volatility']
    
    # Calculate Volume Displacement
    df['volume_displacement'] = df['volume'] - df['volume'].shift(1)
    
    # Combine Intraday Factors
    df['intraday_factors'] = df['intraday_return_adjusted'] * df['volume_displacement']
    
    # Integrate Intraday and Historical Factors
    df['integrated_factor'] = df['integrated_momentum_vol_shock'] * df['intraday_factors']
    
    # Add the result to 20-day Price Momentum
    df['price_momentum_20'] = df['close'] - df['close'].shift(20)
    
    # Add the result to 10-day Volume Acceleration
    df['volume_acceleration_10'] = df['volume'] - df['volume'].shift(10)
    
    # Add the result to 20-day Average True Range
    df['true_range'] = df[['high', 'low', 'close']].join(df['close'].shift(1)).apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close_shift']), abs(x['low'] - x['close_shift'])), axis=1
    )
    df['avg_true_range_20'] = df['true_range'].rolling(window=20).mean()
    
    # Final Factor Combination
    df['final_factor'] = (
        df['integrated_factor'] 
        + df['price_momentum_20'] 
        + df['volume_acceleration_10'] 
        + df['avg_true_range_20']
    )
    
    return df['final_factor']
