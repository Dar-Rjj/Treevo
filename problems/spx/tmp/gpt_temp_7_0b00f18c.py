import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] / df['low']) - 1
    
    # Identify Volume Surge
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_surge'] = (df['volume'] > 2 * df['avg_volume_20']).astype(int)
    
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'] - df['close'].shift(1)
    
    # Adjust Momentum by Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    df['adjusted_momentum_volatility'] = df['daily_momentum'] / df['intraday_volatility']
    
    # Adjust Momentum by Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    df['adjusted_momentum_range'] = df['daily_momentum'] / df['intraday_range']
    
    # Synthesize Combined Indicator
    df['combined_indicator'] = (df['intraday_return'] * df['volume_surge'] + 0.01) * df['adjusted_momentum_range']
    
    # Calculate Price Momentum
    n = 20  # Example lookback period
    df['price_momentum'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    
    # Calculate Volume-Weighted Price Momentum
    df['volume_weighted_price_momentum'] = df['price_momentum'] * df['volume']
    
    # Combine Adjusted Momentum, Price Momentum, and Volume-Weighted Price Momentum
    df['combined_factor'] = df['adjusted_momentum_range'] + df['price_momentum'] + df['volume_weighted_price_momentum']
    
    # Weight by Trade Intensity
    df['vwap'] = (df['amount'] / df['volume'])
    df['average_price'] = (df['high'] + df['low']) / 2
    df['trade_intensity'] = df['volume'] / df['average_price']
    df['weighted_combined_factor'] = df['combined_factor'] * df['trade_intensity']
    
    # Calculate High-to-Low Ratio
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Compute High-to-Low Return
    df['high_low_return'] = df['high_low_ratio'] - 1
    
    # Calculate Exponential Weighted Average of Close Prices
    df['ewm_close'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Intraday Move
    df['intraday_move'] = df['close'] - df['open']
    
    # Adjust Intraday Move by Trade Intensity
    df['weighted_intraday_move'] = df['intraday_move'] * df['trade_intensity']
    
    # Weight Intraday Move and Daily Momentum by Trade Intensity
    df['weighted_daily_momentum'] = df['daily_momentum'] * df['trade_intensity']
    
    # Synthesize Combined Indicator
    df['final_combined_indicator'] = (df['high_low_return'] * df['volume_surge'] + 0.01) * df['ewm_close']
    
    # Identify Volume Spikes
    m = 20  # Example lookback period for volume spike
    df['avg_volume_m'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['avg_volume_m'])
    df['scaled_adjusted_momentum'] = df['adjusted_momentum_range'] * (2 if df['volume_spike'] else 1)
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['final_combined_indicator'] + df['weighted_combined_factor'] + df['price_momentum']
    
    return df['final_alpha_factor']
