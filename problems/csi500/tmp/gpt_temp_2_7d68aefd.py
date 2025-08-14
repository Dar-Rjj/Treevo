import pandas as pd
import pandas as pd

def heuristics_v2(df, short_sma_period=20, long_sma_period=200, vol_lookback=20, momentum_lookback=10, pct_change_lookback=10, turnover_lookback=20):
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_short'] = df['close'].rolling(window=short_sma_period).mean()
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['vol_weighted_high_low_diff'] = df['high_low_diff'] * df['volume']
    df['vol_adjusted_volatility'] = df['vol_weighted_high_low_diff'].rolling(window=vol_lookback).mean()
    
    # Compute Price Momentum
    avg_close = df['close'].rolling(window=momentum_lookback).mean()
    df['price_momentum'] = (df['close'] - df['SMA_short']) / avg_close
    
    # Incorporate Additional Price Change Metrics
    df['pct_change_close'] = df['close'].pct_change(periods=pct_change_lookback)
    df['high_low_range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['SMA_long'] = df['close'].rolling(window=long_sma_period).mean()
    df['market_trend'] = df.apply(lambda row: 'Bullish' if row['SMA_short'] > row['SMA_long'] else 'Bearish', axis=1)
    
    # Incorporate Dynamic Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['rolling_daily_turnover'] = df['daily_turnover'].rolling(window=turnover_lookback).mean()
    
    # Define Weights for Each Component
    base_weights = {
        'price_momentum': 0.4,
        'vol_adjusted_volatility': -0.3,
        'pct_change_close': 0.2,
        'high_low_range': 0.1
    }
    
    # Dynamically Adjust Weights Based on Market Trend and Liquidity
    def adjust_weights(row):
        weights = base_weights.copy()
        if row['market_trend'] == 'Bullish':
            weights['price_momentum'] += 0.1
            weights['vol_adjusted_volatility'] -= 0.1
        else:
            weights['price_momentum'] -= 0.1
            weights['vol_adjusted_volatility'] += 0.1
        
        if row['rolling_daily_turnover'] > df['rolling_daily_turnover'].median():
            weights['price_momentum'] += 0.1
            weights['pct_change_close'] += 0.1
            weights['vol_adjusted_volatility'] -= 0.2
        else:
            weights['price_momentum'] -= 0.1
            weights['pct_change_close'] -= 0.1
            weights['vol_adjusted_volatility'] += 0.2
        
        return weights
    
    # Apply the adjusted weights to create the final alpha factor
    df['alpha_factor'] = df.apply(lambda row: (
        adjust_weights(row)['price_momentum'] * row['price_momentum'] +
        adjust_weights(row)['vol_adjusted_volatility'] * row['vol_adjusted_volatility'] +
        adjust_weights(row)['pct_change_close'] * row['pct_change_close'] +
        adjust_weights(row)['high_low_range'] * row['high_low_range']
    ), axis=1)
    
    # Return the alpha factor as a Series
    return df['alpha_factor']
