import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted High-Low Price Difference
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_weighted_high_low'] = df['high_low_diff'] * df['volume']
    
    # Calculate Momentum
    lookback_period = 10
    df['momentum'] = (df['volume_weighted_high_low'].rolling(window=lookback_period).sum() / 
                      df['volume'].rolling(window=lookback_period).sum())
    
    # Calculate Intraday Return Ratio Components
    df['high_over_low'] = df['high'] / df['low']
    df['close_over_open'] = df['close'] / df['open']
    
    # Evaluate Overnight Sentiment
    df['log_volume'] = np.log(df['volume'])
    df['overnight_return'] = df['open'] / df['close'].shift(1) - 1
    
    # Integrate Intraday and Overnight Signals
    df['intraday_signal'] = (df['high_over_low'] + df['close_over_open']) / 2
    df['intraday_overnight_signal'] = df['intraday_signal'] - df['overnight_return']
    
    # Integrate Historical High-Low Range and Momentum Contributions
    df['high_low_range_5_days'] = df['high_low_diff'].rolling(window=5).sum()
    df['weighted_momentum_contributions'] = (df['volume_weighted_high_low'] * df['momentum']).rolling(window=lookback_period).sum()
    df['integrated_value'] = df['high_low_range_5_days'] * (df['volume_weighted_high_low'] > 0)
    
    # Adjust for Market Sentiment
    df['volatility_threshold'] = (df[['high', 'low']].rolling(window=5).mean().diff(axis=1)['high'] / df['close']).mean()
    df['adjusted_integrated_value'] = np.where(df['integrated_value'] > df['volatility_threshold'], 
                                               df['integrated_value'] + (df['integrated_value'] - df['volatility_threshold']), 
                                               df['integrated_value'] - (df['volatility_threshold'] - df['integrated_value']))
    
    # Integrate Volume Trend and Reversal Potential
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_direction'] = np.where(df['volume_ratio'] > 1, 1, -1)
    df['intraday_high_low_diff'] = df['high'] - df['low']
    df['daily_momentum_change'] = df['close'] - df['close'].shift(1)
    df['reversal_potential'] = df['intraday_high_low_diff'] * df['volume_direction'] - df['daily_momentum_change']
    
    # Generate Composite Indicator
    df['composite_indicator'] = (df['momentum'] + 
                                 df['intraday_overnight_signal'] + 
                                 df['adjusted_integrated_value'] + 
                                 df['reversal_potential'])
    
    # Incorporate Volume-Weighted Intraday Sentiment
    df['volume_weighted_intraday_return'] = (df['close'] - df['open']) * df['volume']
    df['final_alpha_factor'] = df['composite_indicator'] * df['volume_weighted_intraday_return']
    
    return df['final_alpha_factor'].dropna()
