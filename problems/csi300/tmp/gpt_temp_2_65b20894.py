import pandas as pd
import pandas as pd

def heuristics_v2(df, window=20, short_window=5, long_window=20, trend_window=10):
    # Compute Daily Gain or Loss
    df['daily_gain_loss'] = df['close'].diff()
    
    # Volume and Price Adjusted Gain/Loss
    df['volume_price_adjusted_gain_loss'] = df['daily_gain_loss'] * df['volume'] * df['close']
    
    # Cumulate Volume and Price Adjusted Value Over Window
    df['cumulative_volume_price_adjusted'] = df['volume_price_adjusted_gain_loss'].rolling(window=window).sum()
    
    # Calculate Simple Moving Averages (SMA) of Close Prices
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()
    
    # Compute Momentum Difference
    df['momentum_diff'] = (df['sma_short'] - df['sma_long']).abs()
    
    # Calculate Realized Volatility
    df['realized_volatility'] = df['close'].pct_change().rolling(window=window).std()
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap_numerator = (df['close'] * df['volume']).rolling(window=window).sum()
    vwap_denominator = df['volume'].rolling(window=window).sum()
    df['vwap'] = vwap_numerator / vwap_denominator
    
    # Compute Adjusted Momentum
    df['adjusted_momentum'] = df['momentum_diff'] * df['vwap'] * df['momentum_diff'].apply(lambda x: 1 if x > 0 else -1)
    
    # Calculate Intraday Price Range Change
    df['intraday_range_change'] = (df['high'] - df['low']).diff()
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'].diff()
    
    # Combine Range and Volume Change
    df['combined_range_volume_change'] = df['intraday_range_change'] + df['volume_change']
    
    # Apply Sign Function for Directional Bias
    df['directional_bias'] = df['combined_range_volume_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Construct Volatility Adjustment
    df['volatility_adjustment'] = 1 / df['realized_volatility']
    
    # Combine Adjusted Momentum and Volatility
    df['combined_momentum_volatility'] = df['adjusted_momentum'] * df['volatility_adjustment']
    
    # Integrate Combined Alpha Factor
    df['combined_alpha_factor'] = df['combined_momentum_volatility'] * df['directional_bias'].replace(0, df['adjusted_momentum'].apply(lambda x: 1 if x > 0 else -1))
    
    # Add Trend Following Component
    df['trend_sma'] = df['close'].rolling(window=trend_window).mean()
    df['trend_signal'] = (df['close'] > df['trend_sma']).astype(int)
    
    # Combine Final Alpha Factor with Trend Signal
    df['final_alpha_factor'] = df['combined_alpha_factor'] * df['trend_signal']
    
    return df['final_alpha_factor']
