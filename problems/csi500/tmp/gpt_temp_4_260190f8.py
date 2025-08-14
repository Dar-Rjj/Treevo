import pandas as pd
import pandas as pd

def heuristics_v2(df, short_sma_period=20, long_sma_period=200, vol_lookback=20, momentum_lookback=10, pct_change_lookback=5, turnover_lookback=20):
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_short'] = df['close'].rolling(window=short_sma_period).mean()
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['vol_weighted_high_low'] = df['high_low_diff'] * df['volume']
    df['vol_adjusted_volatility'] = df['vol_weighted_high_low'].rolling(window=vol_lookback).mean()
    
    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['SMA_short']) / df['close'].rolling(window=momentum_lookback).mean()
    
    # Incorporate Additional Price Change Metrics
    df['pct_change_close'] = df['close'].pct_change(periods=pct_change_lookback)
    df['high_low_range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['SMA_long'] = df['close'].rolling(window=long_sma_period).mean()
    df['trend_indicator'] = (df['SMA_short'] > df['SMA_long']).astype(int)
    
    # Incorporate Dynamic Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['turnover_rolling_avg'] = df['daily_turnover'].rolling(window=turnover_lookback).mean()
    liquidity_weight = df['turnover_rolling_avg'].rank(pct=True)
    
    # Final Alpha Factor
    weights = {
        'price_momentum': 0.4,
        'vol_adjusted_volatility': -0.2,
        'pct_change_close': 0.1,
        'high_low_range': 0.1,
        'liquidity_weight': 0.2
    }
    
    # Adjust Weights Dynamically Based on Trend
    bullish_weights = {
        'price_momentum': 0.5,
        'vol_adjusted_volatility': -0.2,
        'pct_change_close': 0.1,
        'high_low_range': 0.1,
        'liquidity_weight': 0.2
    }
    
    bearish_weights = {
        'price_momentum': 0.3,
        'vol_adjusted_volatility': -0.2,
        'pct_change_close': 0.2,
        'high_low_range': 0.1,
        'liquidity_weight': 0.2
    }
    
    def adjust_weights(row):
        if row['trend_indicator']:
            return sum([row[k] * v for k, v in bullish_weights.items()])
        else:
            return sum([row[k] * v for k, v in bearish_weights.items()])
    
    df['alpha_factor'] = df.apply(adjust_weights, axis=1)
    
    return df['alpha_factor']

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
