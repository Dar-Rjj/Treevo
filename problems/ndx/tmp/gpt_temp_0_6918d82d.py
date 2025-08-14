import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Compute Volume-Weighted High-Low Price Difference
    df['volume_weighted_high_low'] = (df['high'] - df['low']) * df['volume']
    
    # Compute Price Change
    df['price_change'] = df['close'].diff()
    
    # Incorporate Volume Impact Factor
    df['volume_impact'] = df['volume'] * df['price_change'].abs()
    
    # Integrate Historical High-Low Range and Momentum Contributions
    df['daily_vol_wt_high_low_range'] = df['volume_weighted_high_low'].rolling(window=5).sum()
    df['momentum_contribution'] = df['price_change'].shift(1) * df['volume_weighted_high_low']
    df['weighted_momentum'] = (df['momentum_contribution'].rolling(window=10).sum() 
                                * (df['price_change'].rolling(window=5).sum() > 0))
    
    # Adjust for Market Sentiment
    df['volatility'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['sentiment_adjusted_value'] = df['weighted_momentum'].apply(lambda x: x * 1.1 if x > df['volatility'] else x * 0.9)
    
    # Evaluate Overnight Sentiment
    df['log_volume'] = np.log(df['volume'])
    df['overnight_return'] = np.log(df['open'] / df['close'].shift(1))
    
    # Integrate Intraday and Overnight Signals
    df['intraday_return'] = (df['high'] / df['low'] + df['close'] / df['open']) / 2 - 1
    df['avg_intraday_overnight_diff'] = df['intraday_return'] - df['overnight_return']
    df['volume_moving_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_adjusted_indicator'] = df['volume_moving_avg'] - df['volume']
    
    # Integrate Volume Trend and Reversal Potential
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_direction'] = df['volume_ratio'].apply(lambda x: 1 if x > 1 else -1)
    df['intraday_high_low_diff'] = df['high'] - df['low']
    df['reversal_potential'] = df['intraday_high_low_diff'] * df['volume_direction'] - df['price_change']
    
    # Synthesize Overall Alpha Factor
    df['alpha_factor'] = (
        df['sentiment_adjusted_value'] + 
        df['avg_intraday_overnight_diff'] * df['volume_adjusted_indicator'] + 
        df['reversal_potential'] * df['volume_adjusted_indicator']
    )
    
    return df['alpha_factor']
