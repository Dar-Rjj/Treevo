import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-Weighted by Volume
    df['close_weighted_by_volume'] = df['close'] * df['volume']
    
    # Calculate Short-Term (5 days) and Long-Term (20 days) Volume-Weighted MAs
    df['short_term_ma'] = df['close_weighted_by_volume'].rolling(window=5).mean() / df['volume'].rolling(window=5).mean()
    df['long_term_ma'] = df['close_weighted_by_volume'].rolling(window=20).mean() / df['volume'].rolling(window=20).mean()
    
    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Compute Rolling Sum Returns over 10, 20, and 50 days
    df['sum_returns_10'] = df['daily_return'].rolling(window=10).sum()
    df['sum_returns_20'] = df['daily_return'].rolling(window=20).sum()
    df['sum_returns_50'] = df['daily_return'].rolling(window=50).sum()
    
    # Calculate Weighted Price Movement
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()
    df['weighted_price_movement_10'] = df['sum_returns_10'] * df['avg_volume_10']
    
    # Subtract Lagged Momentum Value
    df['lagged_momentum_10'] = df['sum_returns_10'].shift(1)
    df['lagged_adjustment_10'] = df['weighted_price_movement_10'] - df['lagged_momentum_10']
    
    # Adjust for Price Volatility
    df['daily_price_range'] = df['high'] - df['low']
    df['avg_daily_price_range_10'] = df['daily_price_range'].rolling(window=10).mean()
    df['volatility_adjusted_weighted_price_movement_10'] = df['lagged_adjustment_10'] / df['avg_daily_price_range_10']
    
    # Calculate Volume-Weighted Momentum
    df['volume_weighted_momentum'] = (df['daily_return'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Calculate Volume Shock Indicator
    df['volume_shock_indicator'] = df['volume'] > 2 * df['volume'].rolling(window=20).mean()
    
    # Adjust Volume-Weighted Momentum
    adjustment_factor = 1.5  # Fixed factor for adjustment
    df['adjusted_volume_weighted_momentum'] = df['volume_weighted_momentum'] + df['volume_shock_indicator'] * adjustment_factor * df['volume_weighted_momentum']
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Historical Price Volatility
    df['historical_volatility'] = df['close'].rolling(window=20).std()
    
    # Calculate Volume Direction
    df['volume_direction'] = (df['volume'] - df['volume'].shift(1)).apply(lambda x: 1 if x > 0 else -1)
    
    # Combine Price Movement and Volume Direction
    df['combined_price_movement'] = df['daily_return'] * df['volume_direction']
    
    # Weight by Volume, Inverse Volatility, and Adjusted Momentum
    df['inverse_historical_volatility'] = 1 / df['historical_volatility']
    df['combined_weights'] = df['volume'] * df['inverse_historical_volatility']
    df['weighted_combined_price_movement'] = df['combined_prices_movement'] * df['combined_weights']
    
    # Final Adjustment
    df['final_adjustment'] = df['weighted_combined_price_movement'] - df['lagged_momentum_10'] + df['adjusted_volume_weighted_momentum']
    
    # Calculate Volume-to-Price Ratio
    df['volume_to_price_ratio'] = df['volume'] / df['close']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['final_adjustment'] * df['intraday_volatility'] * df['volume_to_price_ratio']
                                - (df['close'] - df['open']).pow(2))
    
    # Compute Moving Average of the final alpha factor
    df['final_alpha_factor'] = df['final_alpha_factor'].rolling(window=20).mean()
    
    return df['final_alpha_factor']
