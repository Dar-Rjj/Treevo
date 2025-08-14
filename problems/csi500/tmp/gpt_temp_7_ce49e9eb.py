import numpy as np
def heuristics_v2(df):
    # Calculate Simple Moving Average (SMA) of Close Prices
    sma_period = 20
    df['SMA'] = df['close'].rolling(window=sma_period).mean()
    
    # Compute Refined Volume-Adjusted Volatility
    high_low_diff = df['high'] - df['low']
    volume_weighted_diff = high_low_diff * df['volume']
    refined_volatility_period = 10
    df['Refined_Volatility'] = volume_weighted_diff.rolling(window=refined_volatility_period).mean()
    
    # Enhance Volatility Measures with ATR
    df['True_Range'] = np.maximum.reduce([df['high'] - df['low'], 
                                         abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))])
    atr_period = 14
    df['ATR'] = df['True_Range'].rolling(window=atr_period).mean()
    
    # Compute Price Momentum
    momentum_period = 5
    df['Momentum'] = (df['close'] - df['SMA']) / df['close'].rolling(window=momentum_period).mean()
    
    # Incorporate Additional Price Change Metrics
    pct_change_period = 1
    df['Pct_Change'] = df['close'].pct_change(periods=pct_change_period)
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    long_sma_period = 100
    df['Long_SMA'] = df['close'].rolling(window=long_sma_period).mean()
    df['Trend_Indicator'] = np.where(df['SMA'] > df['Long_SMA'], 1, -1)
    
    # Incorporate Dynamic Liquidity Measures
    df['Daily_Turnover'] = df['volume'] * df['close']
    liquidity_period = 30
    df['Liquidity'] = df['Daily_Turnover'].rolling(window=liquidity_period).mean()
    
    # Define Weights
    weights_bullish = {'Momentum': 0.5, 'Refined_Volatility': 0.25, 'Pct_Change': 0.15, 'Trend_Indicator': 0.05, 'Liquidity': 0.05}
    weights_bearish = {'Momentum': 0.3, 'Refined_Volatility': 0.4, 'Pct_Change': 0.15, 'Trend_Indicator': 0.05, 'Liquidity': 0.05}
    weights_high_liquidity = {'Momentum': 0.5, 'Refined_Volatility': 0.25, 'Pct_Change': 0.1, 'Trend_Indicator': 0.05, 'Liquidity': 0.1}
    weights_low_liquidity = {'Momentum': 0.3, 'Refined_Volatility': 0.4, 'Pct_Change': 0.15, 'Trend_Indicator': 0.05, 'Liquidity': 0.05}
    
    # Combine the Components
    def combine_factors(row):
        if row['Trend_Indicator'] == 1 and row['Liquidity'] > df['Liquidity'].median():
            return (row['Momentum'] * weights_bullish['Momentum'] +
