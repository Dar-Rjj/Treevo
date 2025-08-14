import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # 5-Day Log Return
    df['log_return_5d'] = np.log(df['close'] / df['close'].shift(5))
    
    # 5-Day Volume Change
    df['volume_change_5d'] = df['volume'] / df['volume'].shift(5)
    
    # Volume-Adjusted Price Momentum
    df['volume_adjusted_momentum'] = df['log_return_5d'] * df['volume_change_5d']
    
    # Intraday and Overnight Dynamics
    df['intraday_high_low'] = (df['high'] / df['low']) - 1
    df['intraday_close_open'] = (df['close'] / df['open']) - 1
    df['overnight_sentiment'] = (df['open'] / df['close'].shift(1)) - 1
    
    # Median of (High/Low, Close/Open)
    df['median_intraday'] = df[['intraday_high_low', 'intraday_close_open']].median(axis=1)
    
    # Subtract Overnight from Median Intraday
    df['intraday_overnight_diff'] = df['median_intraday'] - df['overnight_sentiment']
    
    # Integrate Intraday and Overnight Signals
    df['average_intraday'] = df[['intraday_high_low', 'intraday_close_open']].mean(axis=1)
    df['recent_volume'] = df['volume']
    df['volume_ma'] = df['volume'].rolling(window=10).mean()
    df['volume_adjusted_indicator'] = df['recent_volume'] - df['volume_ma']
    
    # Synthesize Momentum and Reversal
    df['synthesized_momentum'] = (df['average_intraday'] - df['overnight_sentiment']) * df['volume_adjusted_momentum']
    
    # Volume-Weighted High-Low
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_volume_weighted'] = df['high_low_range'] * df['volume']
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['high_low_volume_ratio'] = df['high_low_range'] / df['volume']
    df['adjusted_alpha_factor'] = df['price_change'] * df['high_low_volume_ratio']
    
    # Composite Momentum
    df['composite_momentum'] = df['high_low_volume_weighted'].rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    df['volume_direction'] = df['volume'] / df['volume'].shift(1)
    df['intraday_high_low_diff'] = df['high'] - df['low']
    df['integrated_intraday_overnight_signal'] = df['composite_momentum'] * df['volume_direction'] * df['intraday_high_low_diff']
    
    # Deviation of VWAP from Close
    df['vwap'] = (df['amount'] / df['volume'])
    df['deviation_vwap_close'] = df['vwap'] - df['close']
    
    # Incorporate Volume Impact Factor
    df['volume_impact_factor'] = df['volume'] * np.abs(df['close'] - df['close'].shift(1))
    
    # Integrate Historical High-Low Range and Momentum Contributions
    df['daily_vol_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    df['daily_momentum_contributions'] = df['log_return_5d'] * df['volume']
    df['positive_price_change'] = df['price_change'].rolling(window=5).apply(lambda x: all(x > 0), raw=True)
    df['weighted_contributions'] = df['daily_vol_weighted_high_low_range'] * df['daily_momentum_contributions']
    df['accumulated_weighted_contributions'] = df['weighted_contributions'].rolling(window=5).sum().where(df['positive_price_change'], 0)
    
    # Generate Alpha Factor
    df['alpha_factor'] = (df['adjusted_alpha_factor'] + 
                          df['deviation_vwap_close'] + 
                          df['synthesized_momentum'] + 
                          df['integrated_intraday_overnight_signal'] + 
                          df['accumulated_weighted_contributions'])
    df['alpha_factor'] = df['alpha_factor'] * df['close']
    
    return df['alpha_factor']
