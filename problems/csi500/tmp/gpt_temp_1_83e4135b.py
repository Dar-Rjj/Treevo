import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate short-term and long-term moving averages of log returns
    df['ma5_log_return'] = df['log_return'].rolling(window=5).mean()
    df['ma20_log_return'] = df['log_return'].rolling(window=20).mean()
    
    # Calculate momentum factor
    df['momentum_factor'] = df['ma5_log_return'] - df['ma20_log_return']
    
    # Calculate historical volatility
    df['volatility_20d'] = df['log_return'].rolling(window=20).std()
    
    # Calculate exponential weighted moving average of log returns
    df['ewma12_log_return'] = df['log_return'].ewm(span=12, adjust=False).mean()
    
    # Price patterns: Bullish and Bearish Engulfing
    df['bullish_engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    df['bearish_engulfing'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    
    # Volume changes
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    
    # Moving Averages of Volume
    df['ma5_volume'] = df['volume'].rolling(window=5).mean()
    df['ma20_volume'] = df['volume'].rolling(window=20).mean()
    
    # Volume Momentum
    df['volume_momentum'] = df['ma5_volume'] - df['ma20_volume']
    
    # Volume Spike
    df['volume_spike'] = df['volume'] > 2 * df['volume'].shift(1)
    
    # Dollar Volume
    df['dollar_volume'] = df['close'] * df['volume']
    
    # Percentage change in dollar volume
    df['dollar_volume_change'] = df['dollar_volume'] / df['dollar_volume'].shift(1) - 1
    
    # Dollar Volume Spike
    df['dollar_volume_spike'] = df['dollar_volume'] > 2 * df['dollar_volume'].shift(1)
    
    # Combined Momentum
    combined_momentum = 0.4 * df['momentum_factor'] + 0.3 * df['volume_momentum'] + 0.3 * df['dollar_volume_change']
    
    # Directional Movement Index (DMI)
    df['dm_plus'] = np.where(df['high'].diff() > 0, df['high'].diff(), 0)
    df['dm_minus'] = np.where(df['low'].diff() < 0, -df['low'].diff(), 0)
    df['tr'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    
    df['smoothed_dm_plus'] = df['dm_plus'].rolling(window=14).sum()
    df['smoothed_dm_minus'] = df['dm_minus'].rolling(window=14).sum()
    df['smoothed_tr'] = df['tr'].rolling(window=14).sum()
    
    df['di_plus'] = 100 * df['smoothed_dm_plus'] / df['smoothed_tr']
    df['di_minus'] = 100 * df['smoothed_dm_minus'] / df['smoothed_tr']
    df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # Composite Factors
    trend_strength = df['adx']
    combined_trend_and_momentum = 0.5 * trend_strength + 0.5 * combined_momentum
    price_volume_patterns = (df['bullish_engulfing'] & df['volume_spike']) | (df['bearish_engulfing'] & df['volume_spike'])
    
    # Final Alpha Factor
    alpha_factor = 0.6 * combined_trend_and_momentum + 0.4 * price_volume_patterns
    
    return alpha_factor
