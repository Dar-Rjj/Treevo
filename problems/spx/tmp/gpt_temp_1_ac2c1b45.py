import pandas as pd
import pandas as pd

def heuristics(df, m=20, n=10):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'] - df['close'].shift(1)
    
    # Adjust Daily Momentum by Intraday Volatility
    df['adjusted_daily_momentum'] = df['daily_momentum'] / df['intraday_volatility']
    
    # Identify Volume Spikes
    df['avg_volume'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] > df['avg_volume']).astype(int)
    
    # Adjust Daily Momentum by Volume Spike
    scaling_factor = 1.5
    df['final_adjusted_momentum'] = df['adjusted_daily_momentum'] * (scaling_factor if df['volume_spike'] == 1 else 1)
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'].pct_change(periods=n)
    
    # Synthesize Combined Indicator
    df['combined_indicator'] = df['final_adjusted_momentum'] * df['price_momentum']
    
    # Calculate High-to-Low Ratio
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Compute High-to-Low Return
    df['high_low_return'] = df['high_low_ratio'] - 1
    
    # Adjust High-to-Low Return by Volume Surge
    volume_surge_threshold = 2 * df['volume'].rolling(window=m).mean()
    df['volume_surge_adjustment'] = (df['volume'] > volume_surge_threshold).astype(int) * 1 + (df['volume'] <= volume_surge_threshold).astype(int) * 0.5
    df['adjusted_high_low_return'] = df['high_low_return'] * df['volume_surge_adjustment']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_indicator'] + df['adjusted_high_low_return']
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 101, 102, 103, 104],
#     'high': [105, 106, 107, 108, 109],
#     'low': [95, 96, 97, 98, 99],
#     'close': [100, 101, 102, 103, 104],
#     'volume': [1000, 1200, 1500, 1800, 2000]
# })
# alpha_factor = heuristics(df)
# print(alpha_factor)
