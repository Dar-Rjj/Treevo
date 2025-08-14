import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Compute Rolling Sum Returns
    N = 20  # Example window
    df['rolling_sum_returns'] = df['daily_return'].rolling(window=N).sum()
    
    # Calculate Weighted Price Movement
    df['avg_volume'] = df['volume'].rolling(window=N).mean()
    df['weighted_price_movement'] = df['rolling_sum_returns'] * df['avg_volume']
    
    # Subtract a Lagged Momentum Value
    df['lagged_rolling_sum_returns'] = df['rolling_sum_returns'].shift(N)
    df['lagged_weighted_price_movement'] = (df['rolling_sum_returns'] - df['lagged_rolling_sum_returns']) * df['avg_volume']
    
    # Incorporate Daily Volume Changes
    df['vol_percentage_change'] = df['volume'].pct_change()
    
    # Aggregate Volume Impact
    df['volume_impact'] = (df['vol_percentage_change'] * df['close']).rolling(window=N).sum() / df['close'].rolling(window=N).sum()
    
    # Calculate Historical Price Volatility
    df['price_volatility'] = df['close'].rolling(window=20).std()
    
    # Inverse Volatility
    df['inverse_volatility'] = 1 / df['price_volatility']
    
    # Combine Weights
    df['combined_weights'] = df['volume_impact'] * df['inverse_volatility']
    df['weighted_price_change'] = df['combined_weights'] * df['daily_return']
    
    # Final Adjustment
    df['final_factor'] = (df['weighted_price_change'] + df['lagged_weighted_price_movement'])
    
    # Analyze the relationship between current close price and moving averages
    df['5_day_sma'] = df['close'].rolling(window=5).mean()
    df['close_to_sma_diff'] = df['close'] - df['5_day_sma']
    
    # Investigate the volatility of daily returns to identify potential momentum
    df['log_returns'] = df['close'].apply(lambda x: np.log(x) - np.log(df['close'].shift(1)))
    df['log_returns_volatility'] = df['log_returns'].rolling(window=20).std()
    
    # Examine the behavior of trading volume in relation to price changes
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['abs_price_change'] = df['close'].diff().abs()
    df['volume_impact_adjusted'] = df['volume_ratio'] * df['abs_price_change']
    
    # Evaluate the impact of intraday price movements on future returns
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_range_pct'] = df['high_low_range'] / df['open']
    df['next_day_return'] = df['close'].shift(-1).pct_change()
    df['high_low_range_corr'] = df['high_low_range_pct'].rolling(window=20).corr(df['next_day_return'])
    
    # Incorporate amount traded into the analysis for a more comprehensive view
    df['amount_ratio'] = df['amount'] / df['amount'].rolling(window=30).mean()
    df['composite_liquidity'] = df['amount_ratio'] * df['volume_ratio']
    
    # Introduce High-Low Range Indicator
    M = 10  # Example window
    df['rolling_high_low_range'] = df['high_low_range'].rolling(window=M).mean()
    df['normalized_high_low_range'] = df['high_low_range'] / df['rolling_high_low_range']
    
    # Incorporate High-Low Range into Final Factor
    df['final_factor'] = df['final_factor'] * df['normalized_high_low_range']
    
    return df['final_factor']
