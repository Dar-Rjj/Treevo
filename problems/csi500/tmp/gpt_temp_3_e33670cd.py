import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Logarithmic Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Close-Open Spread
    df['close_open_spread'] = df['close'] - df['open']
    
    # Volume Weighted High-Low Spread
    df['volume_weighted_high_low_spread'] = (df['high'] - df['low']) * df['volume']
    
    # Enhanced Liquidity Measure
    df['enhanced_liquidity_measure'] = df['volume'] / df['amount']
    
    # 20-Day Simple Moving Average of Close Price
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Market Trend Adjustment
    df['market_trend_adjustment'] = df['close'] - df['sma_20']
    
    # Refined Momentum Calculation
    df['refined_momentum'] = df['log_return'].rolling(window=5).mean() * (df['volume'] / df['volume'].rolling(window=5).mean())
    
    # Combine Intraday Volatility, Momentum, Volume-Weighted, Enhanced Liquidity, and Market Trend Measures
    df['combined_measures'] = df['intraday_high_low_spread'] + df['refined_momentum'] + df['volume_weighted_high_low_spread'] + df['enhanced_liquidity_measure'] + df['market_trend_adjustment']
    
    # Gap Adjustment
    df['gap_difference'] = df.apply(lambda row: row['open'] - df.loc[row.name - pd.Timedelta(days=1), 'close'] if row['open'] > df.loc[row.name - pd.Timedelta(days=1), 'close'] else df.loc[row.name - pd.Timedelta(days=1), 'close'] - row['open'], axis=1)
    df['adjusted_gap_difference'] = df['gap_difference'] * df['volume']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_measures'] + df['adjusted_gap_difference']
    
    return df['alpha_factor'].dropna()
