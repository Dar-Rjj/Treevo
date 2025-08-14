import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Combined Momentum
    df['1d_return'] = df['close'].pct_change()
    df['5d_return'] = df['close'].pct_change(5)
    df['20d_return'] = df['close'].pct_change(20)
    df['3d_return'] = df['close'].pct_change(3)
    df['10d_return'] = df['close'].pct_change(10)
    df['30d_return'] = df['close'].pct_change(30)
    df['average_momentum'] = (df['1d_return'] + df['5d_return'] + df['20d_return'] + 
                              df['3d_return'] + df['10d_return'] + df['30d_return']) / 6
    
    # Filter by Relative Volume
    n_days = 20  # Number of days to calculate average volume
    df['avg_volume'] = df['volume'].rolling(window=n_days).mean()
    df_filtered = df[df['volume'] > df['avg_volume']]
    
    # Adjusted High-Low Cumulative Return and Weighted Price Gaps
    df_filtered['daily_price_range'] = df_filtered['high'] - df_filtered['low']
    df_filtered['sum_daily_price_ranges'] = df_filtered['daily_price_range'].sum()
    df_filtered['volume_growth'] = df_filtered['volume'].pct_change()
    df_filtered['weighted_price_gaps'] = df_filtered['daily_price_range'] * df_filtered['volume_growth']
    sum_weighted_price_gaps = df_filtered['weighted_price_gaps'].sum()
    
    # High-to-Low Range and Price Change Synthesis
    df['high_low_range'] = df['high'] - df['low']
    df['price_change'] = df['close'] - df['open'].shift(1)
    df['combined_value'] = df['high_low_range'] * 150 + df['price_change']
    
    # Smoothing and Initial Momentum Calculation
    df['sma_7'] = df['combined_value'].rolling(window=7).mean()
    df['initial_momentum'] = df['sma_7'] - df['sma_7'].shift(7)
    
    # Adjust for Volume Changes
    df['daily_vol_pct_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['final_vol_adjustment'] = df['average_momentum'] * (1 + 2 * df['daily_vol_pct_change'])
    
    # Intermediate Factor Combination
    df['intermediate_factor'] = df['initial_momentum'] + df['final_vol_adjustment']
    
    # Calculate Weighted Price Momentum
    df['log_return'] = df['close'].apply(lambda x: math.log(x))
    df['weighted_price_momentum'] = (df['close'] - df['close'].rolling(window=15).mean()) * df['log_return']
    
    # Generate Final Alpha Factor
    df['final_alpha_factor'] = (df['sum_daily_price_ranges'] + sum_weighted_price_gaps) * df['average_momentum'] + \
                               df['intermediate_factor'] + df['weighted_price_momentum']
    
    return df['final_alpha_factor']
