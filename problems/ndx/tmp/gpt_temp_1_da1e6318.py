import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Gap
    df['Price_Gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate Momentum
    def calculate_momentum(close, n):
        sma = close.rolling(window=n).mean()
        price_diff = close - sma
        momentum_score = price_diff / sma
        return momentum_score
    
    df['Momentum_5'] = calculate_momentum(df['close'], 5)
    df['Momentum_10'] = calculate_momentum(df['close'], 10)
    
    # Weight by Volume
    df['Volume_Adjusted_Momentum_5'] = df['Momentum_5'] / df['volume']
    df['Volume_Adjusted_Momentum_10'] = df['Momentum_10'] / df['volume']
    
    df['Weighted_Momentum_5'] = df['Price_Gap'] * df['Volume_Adjusted_Momentum_5']
    df['Weighted_Momentum_10'] = df['Price_Gap'] * df['Volume_Adjusted_Momentum_10']
    
    # Volume-Weighted Factor
    df['Daily_Returns'] = df['high'] - df['close'].shift(1)
    
    def volume_weighted_factor(daily_returns, volume, n):
        aggregate_product = (daily_returns * volume).rolling(window=n).sum()
        aggregate_volume = volume.rolling(window=n).sum()
        final_volume_weighted_factor = aggregate_product / aggregate_volume
        return final_volume_weighted_factor
    
    df['Volume_Weighted_Factor_5'] = volume_weighted_factor(df['Daily_Returns'], df['volume'], 5)
    df['Volume_Weighted_Factor_10'] = volume_weighted_factor(df['Daily_Returns'], df['volume'], 10)
    
    # Integrate
    df['Integrated_Factor_5'] = df['Momentum_5'] * df['Volume_Weighted_Factor_5']
    df['Integrated_Factor_10'] = df['Momentum_10'] * df['Volume_Weighted_Factor_10']
    
    # Output the final integrated factor
    return df['Integrated_Factor_5'], df['Integrated_Factor_10']
