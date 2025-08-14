import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Define lookback periods
    sma_short_period = 20
    sma_long_period = 50
    vol_adj_volatility_period = 20
    momentum_lookback = 20
    pct_change_period = 5
    liquidity_lookback = 20
    vwap_lookback = 20
    
    # Weights for the final alpha factor
    weights = {
        'price_momentum': 0.3,
        'vol_adj_volatility': 0.15,
        'pct_change': 0.1,
        'high_low_range': 0.05,
        'market_trend': 0.1,
        'liquidity_score': 0.3
    }
    
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['sma_short'] = df['close'].rolling(window=sma_short_period).mean()
    df['sma_long'] = df['close'].rolling(window=sma_long_period).mean()
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['vol_weighted_high_low_diff'] = df['high_low_diff'] * df['volume']
    df['vol_adj_volatility'] = df['vol_weighted_high_low_diff'].rolling(window=vol_adj_volatility_period).mean()
    
    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['sma_short']) / df['close'].rolling(window=momentum_lookback).mean()
    
    # Incorporate Additional Price Change Metrics
    df['pct_change'] = df['close'].pct_change(periods=pct_change_period)
    df['high_low_range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['market_trend'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)
    
    # Incorporate Enhanced Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['avg_turnover'] = df['daily_turnover'].rolling(window=liquidity_lookback).mean()
    df['liquidity_ratio'] = df['daily_turnover'] / df['avg_turnover']
    
    df['vwap'] = (df['volume'] * df['close']).rolling(window=vwap_lookback).sum() / df['volume'].rolling(window=vwap_lookback).sum()
    df['liquidity_score'] = (weights['daily_turnover'] * df['daily_turnover'] + 
                             weights['avg_turnover'] * df['avg_turnover'] + 
                             weights['liquidity_ratio'] * df['liquidity_ratio']) / 3
    
    # Dynamically Adjust Weights Based on Market Trend
    df['final_alpha_factor'] = (weights['price_momentum'] * df['price_momentum'] +
                                weights['vol_adj_volatility'] * df['vol_adj_volatility'] +
                                weights['pct_change'] * df['pct_change'] +
                                weights['high_low_range'] * df['high_low_range'] +
                                weights['market_trend'] * df['market_trend'] +
                                weights['liquidity_score'] * df['liquidity_score'])
    
    return df['final_alpha_factor']
