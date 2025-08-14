import pandas as pd
import pandas as pd

def heuristics_v2(df, n=21):
    # Calculate Volume-Weighted Average Prices
    df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume_weighted_avg_price'] = df['avg_price'] * df['volume']
    
    # Calculate Volume-Weighted Moving Average
    rolling_volume = df['volume'].rolling(window=n).sum()
    df['volume_weighted_moving_avg'] = df['volume_weighted_avg_price'].rolling(window=n).sum() / rolling_volume
    
    # Compute Current Day's Volume-Weighted Price
    df['current_day_vol_weighted_price'] = df['avg_price'] * df['volume']
    
    # Calculate VWPTI
    df['VWPTI'] = (df['current_day_vol_weighted_price'] - df['volume_weighted_moving_avg']) / df['volume_weighted_moving_avg']
    
    # Calculate Daily Price Return
    df['daily_price_return'] = df['close'] / df['close'].shift(1) - 1
    
    # Calculate Volume Shock Factor
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_21_ma'] = df['volume'].rolling(window=21).mean()
    df['volume_shock_factor'] = df['volume_change'] / df['volume_21_ma']
    
    # Adjust with Open and Close Spread
    df['open_close_spread'] = df['close'] - df['open']
    
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] / df['open'] - 1
    
    # Adjust for Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
    
    # Volume Trend Factor
    df['volume_5_ma'] = df['volume'].rolling(window=5).mean()
    df['volume_trend_factor'] = (df['volume'] - df['volume_5_ma']) * df['intraday_return']
    
    # Amount Momentum Factor
    df['amount'] = (df['high'] + df['low']) / 2 * df['volume']
    df['amount_5_ma'] = df['amount'].rolling(window=5).mean()
    df['amount_momentum_factor'] = (df['amount'] - df['amount_5_ma']) * df['intraday_return']
    
    # Integrate Combined Weighted Price Changes
    df['high_low_weighted'] = (df['high'] - df['low']) * df['volume']
    df['open_close_weighted'] = df['open_close_spread'] * df['volume']
    df['combined_weighted_price_changes'] = df['high_low_weighted'] + df['open_close_weighted']
    
    # Identify Directional Days
    df['direction'] = df.apply(lambda row: 'Up' if row['close'] > row['open'] else 'Down', axis=1)
    df['up_count'] = df['direction'].rolling(window=n).apply(lambda x: (x == 'Up').sum(), raw=False)
    df['down_count'] = df['direction'].rolling(window=n).apply(lambda x: (x == 'Down').sum(), raw=False)
    df['directional_days'] = (df['up_count'] - df['down_count']) * df['volume']
    
    # Combine Factors
    df['vwpti_adjusted'] = df['VWPTI'] * df['daily_price_return'] * df['volume_shock_factor']
    
    # Synthesize Components
    df['alpha_factor'] = (
        df['vwpti_adjusted'] +
        df['open_close_spread'] +
        df['combined_weighted_price_changes'] +
        df['directional_days'] +
        df['volume_trend_factor'] +
        df['amount_momentum_factor']
    )
    
    return df['alpha_factor']
