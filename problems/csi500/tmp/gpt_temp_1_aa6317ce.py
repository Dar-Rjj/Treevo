import pandas as pd
import pandas as pd

def heuristics_v2(df, sma_short_period=20, sma_long_period=100, vol_lookback=20, pct_change_lookback=10, turnover_lookback=20):
    # Calculate Simple Moving Average (SMA) of Close Prices
    sma_short = df['close'].rolling(window=sma_short_period).mean()
    
    # Compute Volume-Adjusted Volatility
    high_low_diff = df['high'] - df['low']
    volume_weighted_volatility = (high_low_diff * df['volume']).rolling(window=vol_lookback).mean()
    
    # Compute Price Momentum
    price_momentum = (df['close'] - sma_short) / df['close'].rolling(window=sma_short_period).mean()
    
    # Incorporate Additional Price Change Metrics
    close_pct_change = df['close'].pct_change(pct_change_lookback)
    high_low_range = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    sma_long = df['close'].rolling(window=sma_long_period).mean()
    trend_indicator = (sma_short > sma_long).astype(int)  # 1 for Bullish, 0 for Bearish
    
    # Incorporate Liquidity Measures
    daily_turnover = df['volume'] * df['close']
    rolling_turnover = daily_turnover.rolling(window=turnover_lookback).mean()
    
    # Define the weights for each component
    weight_momentum = 0.4
    weight_volatility = 0.2
    weight_close_pct_change = 0.1
    weight_high_low_range = 0.1
    weight_liquidity = 0.2
    
    # Adjust Weights Based on Market Trend
    weight_momentum_bullish = weight_momentum * 1.5 if trend_indicator else weight_momentum * 0.75
    weight_volatility_bullish = weight_volatility * 0.75 if trend_indicator else weight_volatility * 1.5
    
    # Adjust Weights Based on Liquidity
    liquidity_factor = rolling_turnover / rolling_turnover.mean()
    weight_momentum_liquidity = weight_momentum_bullish * liquidity_factor
    weight_volatility_liquidity = weight_volatility_bullish * liquidity_factor
    weight_close_pct_change_liquidity = weight_close_pct_change * liquidity_factor
    weight_high_low_range_liquidity = weight_high_low_range * liquidity_factor
    weight_liquidity_adjusted = weight_liquidity * liquidity_factor
    
    # Normalize the weights to ensure they sum to 1
    total_weight = (weight_momentum_liquidity + weight_volatility_liquidity + 
                    weight_close_pct_change_liquidity + weight_high_low_range_liquidity + 
                    weight_liquidity_adjusted)
    
    weight_momentum_liquidity /= total_weight
    weight_volatility_liquidity /= total_weight
    weight_close_pct_change_liquidity /= total_weight
    weight_high_low_range_liquidity /= total_weight
    weight_liquidity_adjusted /= total_weight
    
    # Final Alpha Factor
    alpha_factor = (price_momentum * weight_momentum_liquidity +
                    volume_weighted_volatility * weight_volatility_liquidity +
                    close_pct_change * weight_close_pct_change_liquidity +
                    high_low_range * weight_high_low_range_liquidity +
                    rolling_turnover * weight_liquidity_adjusted)
    
    return alpha_factor
