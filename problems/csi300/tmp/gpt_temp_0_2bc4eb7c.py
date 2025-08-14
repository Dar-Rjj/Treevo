import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Range
    df['intraday_high_low_range'] = df['high'] - df['low']
    
    # Calculate Close to Open Difference
    df['close_to_open_diff'] = df['close'] - df['open']
    
    # Compute Intraday Momentum Score
    df['intraday_momentum_score'] = (df['intraday_high_low_range'] + df['close_to_open_diff']) * df['volume']
    
    # Calculate Volume Weighted Price
    df['volume_weighted_price'] = (df['close'] * df['volume'] + df['open'] * df['volume']) / (2 * df['volume'])
    
    # Calculate Volume Weighted Price Change
    df['volume_weighted_price_change'] = df['volume_weighted_price'] - df['volume_weighted_price'].shift(1)
    
    # Compute Combined Intraday and Daily Momentum
    df['combined_momentum'] = df['intraday_momentum_score'] + df['volume_weighted_price_change']
    
    # Apply Exponential Moving Average (EMA) to Combined Momentum
    df['ema_combined_momentum'] = df['combined_momentum'].ewm(span=20, adjust=False).mean()
    
    # Calculate Intraday Volatility Score
    df['high_close_delta'] = df['high'] - df['close']
    df['low_close_delta'] = df['low'] - df['close']
    df['avg_deltas'] = (df['high_close_delta'] + df['low_close_delta']) / 2
    df['intraday_volatility_score'] = df['avg_deltas'] * df['amount']
    
    # Compute EMA of Squared Returns for Volatility
    df['daily_returns'] = df['close'].pct_change()
    df['squared_returns'] = df['daily_returns'] ** 2
    df['ema_squared_returns'] = df['squared_returns'].ewm(span=20, adjust=False).mean()
    
    # Calculate Divergence between EMA of Returns and Volatility
    df['divergence'] = abs(df['ema_combined_momentum'] - df['ema_squared_returns'])
    
    # Calculate Divergence and Weight Factor
    df['divergence_weight_factor'] = (df['divergence'] * df['volume']) / df['intraday_high_low_range']
    
    # Integrate Composite Momentum, Volatility, and Volume Trends
    df['ema_30_combined_momentum'] = df['combined_momentum'].ewm(span=30, adjust=False).mean()
    df['integrated_scores'] = df['combined_momentum'] - df['ema_30_combined_momentum'] - df['intraday_volatility_score']
    
    # Introduce Time-Varying Volatility Adjustment
    df['long_term_volatility'] = df['squared_returns'].ewm(span=50, adjust=False).mean()
    df['short_term_volatility'] = df['squared_returns'].ewm(span=10, adjust=False).mean()
    df['volatility_ratio'] = df['short_term_volatility'] / df['long_term_volatility']
    df['adjusted_integrated_scores'] = df['integrated_scores'] * df['volatility_ratio']
    
    # Incorporate Trend Reversal Indicators
    df['sma_5_close'] = df['close'].rolling(window=5).mean()
    df['sma_20_close'] = df['close'].rolling(window=20).mean()
    df['trend_direction'] = np.where(df['sma_5_close'] > df['sma_20_close'], 1, -1)
    df['final_score'] = df['adjusted_integrated_scores'] * df['trend_direction']
    
    # Enhance Volatility Measure with True Range
    df['true_range'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    df['enhanced_intraday_volatility_score'] = df['intraday_volatility_score'] + (df['true_range'] * df['volume'])
    
    # Final factor
    df['alpha_factor'] = df['final_score'] - df['enhanced_intraday_volatility_score']
    
    return df['alpha_factor']
