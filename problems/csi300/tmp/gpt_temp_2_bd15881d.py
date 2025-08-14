import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Price Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Volume Influence Ratio
    upward_volume = df[df['close'] > df['open']]['volume'].sum()
    downward_volume = df[df['close'] < df['open']]['volume'].sum()
    volume_influence_ratio = (upward_volume / downward_volume).fillna(1)
    
    # Multiply High-Low Difference with Volume Influence Ratio
    high_low_vol_influence = high_low_diff * volume_influence_ratio
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Identify Volume Trend
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_trend_strength = volume_change.apply(lambda x: 1.5 if x > 0 else 0.5)
    
    # Combine High-Low Difference, Volume Influence, and Intraday Return
    combined_vol_influence_return = high_low_vol_influence * intraday_return * volume_trend_strength
    
    # Calculate Daily Price Momentum
    price_momentum = df['close'] - df['close'].shift(10)
    
    # Calculate Volume Surprise
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_surprise = df['volume'] - volume_ma_10
    combined_price_momentum_vol_surprise = price_momentum * volume_surprise
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = combined_vol_influence_return * combined_price_momentum_vol_surprise
    
    return final_alpha_factor
