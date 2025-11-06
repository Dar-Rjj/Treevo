import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['acceleration_factor'] = (data['short_term_momentum'] - data['medium_term_momentum']) * data['medium_term_momentum']
    
    # Intraday Pressure
    data['buying_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1e-10)) * data['volume']
    data['selling_pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low']).replace(0, 1e-10)) * data['volume']
    data['net_pressure_signal'] = (data['buying_pressure'] - data['selling_pressure']) * np.sign(data['short_term_momentum'])
    
    # Volatility Breakout
    data['rolling_max_high'] = data['high'].rolling(window=5, min_periods=1).max()
    data['range_breakout'] = (data['close'] - data['rolling_max_high']) * (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1e-10)
    data['volatility_adjustment'] = data['range_breakout'] / ((data['high'] - data['low']) / ((data['high'] + data['low']) / 2)).replace(0, 1e-10)
    
    # Flow Persistence
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['price_change_sign'] = np.sign(data['price_change'])
    
    # Direction Consistency - count same sign in past 5 days
    def count_same_sign(series):
        if len(series) < 6:
            return np.nan
        current_sign = series.iloc[-1]
        return (series.iloc[-6:-1] == current_sign).sum()
    
    data['direction_consistency'] = data['price_change_sign'].rolling(window=6, min_periods=6).apply(count_same_sign, raw=False)
    
    # Volume Stability - 5-day correlation of volume with lagged volume
    def volume_correlation(series):
        if len(series) < 5:
            return np.nan
        return pd.Series(series).corr(pd.Series(series).shift(1))
    
    data['volume_stability'] = data['volume'].rolling(window=5, min_periods=5).apply(volume_correlation, raw=False)
    
    # Composite Alpha
    data['core_factor'] = data['acceleration_factor'] * data['net_pressure_signal'] * data['volatility_adjustment']
    data['final_alpha'] = data['core_factor'] * data['direction_consistency'] * data['volume_stability']
    
    # Return the final alpha factor series
    return data['final_alpha']
