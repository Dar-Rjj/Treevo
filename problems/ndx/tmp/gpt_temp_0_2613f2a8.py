import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Momentum Contribution with Volume Impact
    df['price_change'] = df['close'].diff()
    df['momentum_contribution'] = df['volume'] * abs(df['price_change'])
    
    # Integrate Historical Momentum Contributions
    df['sum_momentum_contributions'] = df['momentum_contribution'].rolling(window=5).sum()
    df['max_min_price_change'] = df['price_change'].rolling(window=5).max() - df['price_change'].rolling(window=5).min()
    df['positive_slope_emphasis'] = df['sum_momentum_contributions'] * (df['max_min_price_change'] > 0).astype(int)
    
    # Adjust for Market Sentiment Using Volatility Threshold
    df['volatility'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['avmi'] = df['positive_slope_emphasis']
    df['adjusted_avmi'] = df['avmi'].apply(lambda x: x * 1.1 if x > df['volatility'] else x * 0.9)
    
    # Calculate Volume-Weighted High-Low Price Difference
    df['volume_weighted_high_low'] = (df['high'] - df['low']) * df['volume']
    
    # Calculate Momentum
    sum_volume_weighted_high_low = df['volume_weighted_high_low'].rolling(window=10).sum()
    sum_volume = df['volume'].rolling(window=10).sum()
    df['momentum'] = sum_volume_weighted_high_low / sum_volume
    
    # Evaluate Intraday and Overnight Signals
    df['high_over_low'] = df['high'] / df['low']
    df['close_over_open'] = df['close'] / df['open']
    df['log_volume'] = df['volume'].apply(lambda x: max(1, x)).apply(np.log)
    df['overnight_return'] = df['open'] / df['close'].shift(1) - 1
    
    # Integrate Intraday and Overnight Signals
    df['intraday_signal'] = (df['high_over_low'] + df['close_over_open']) / 2
    df['integrated_signal'] = df['intraday_signal'] - df['overnight_return']
    
    # Generate Composite Indicator
    df['composite_indicator'] = df['momentum'] + df['integrated_signal']
    
    # Volume Weighted Adjustment
    df['volume_moving_avg'] = df['volume'].rolling(window=10).mean()
    df['recent_volume_adj'] = df['volume_moving_avg'] - df['volume']
    
    df['composite_indicator'] = df['composite_indicator'] * df['recent_volume_adj']
    
    # Synthesize Overall Alpha Factor
    df['alpha_factor'] = df['composite_indicator']
    
    return df['alpha_factor']
