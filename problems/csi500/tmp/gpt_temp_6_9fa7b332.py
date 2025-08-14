import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 10-Day Sum of High-Low Ranges
    df['10_day_sum_high_low_range'] = df['high_low_range'].rolling(window=10).sum()
    
    # Calculate Price Change over 10 Days
    df['price_change_10_days'] = df['close'] - df['close'].shift(10)
    
    # Calculate Intraday Mean Price
    df['intraday_mean_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Measure Intraday Deviation
    df['intraday_deviation'] = df['high_low_range'] / df['intraday_mean_price']
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Close Momentum
    n = 5  # Number of periods for close momentum
    df['close_momentum'] = sum((df['close'] - df['close'].shift(i)) / df['close'].shift(i) for i in range(1, n+1))
    
    # Apply Enhanced Volume Filter to Price Momentum
    lookback_period = 20
    df['daily_avg_volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['volume_classification'] = pd.cut(df['volume'], bins=[0, df['daily_avg_volume']*0.8, df['daily_avg_volume']*1.2, float('inf')], labels=['Low', 'Medium', 'High'])
    
    # Apply Amount Filter to Price Momentum
    df['daily_avg_amount'] = df['amount'].rolling(window=lookback_period).mean()
    df['amount_classification'] = pd.cut(df['amount'], bins=[0, df['daily_avg_amount']*0.8, df['daily_avg_amount']*1.2, float('inf')], labels=['Low', 'Medium', 'High'])
    
    # Combine Price Momentum with Volume and Amount Classification
    weight_map = {
        ('High', 'High'): 1.5,
        ('High', 'Low'): 1.0,
        ('Low', 'High'): 1.0,
        ('Low', 'Low'): 0.5
    }
    df['combined_weight'] = df.apply(lambda row: weight_map[(row['volume_classification'], row['amount_classification'])], axis=1)
    df['final_price_momentum'] = df['price_momentum'] * df['combined_weight']
    
    # Volume Confirmation
    m = 21  # Number of days for volume trend
    df['volume_trend'] = df['volume'].rolling(window=m).mean()
    df['volume_score'] = (df['volume_trend'] - df['volume_trend'].shift(m)).rolling(window=21).mean()
    
    # Scale 10-Day Sum of High-Low Ranges by Intraday Mean Price
    df['scaled_10_day_sum_high_low_range'] = df['10_day_sum_high_low_range'] / df['intraday_mean_price']
    
    # Integrate All Scores
    df['alpha_factor'] = (df['final_price_momentum'] * df['volume_score'] 
                          + df['scaled_10_day_sum_high_low_range'] 
                          + df['scaled_10_day_sum_high_low_range'] * df['close_momentum'])
    
    return df['alpha_factor']
