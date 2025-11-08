import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Momentum Components
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['range_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1
    
    # Microstructure Signals
    data['intraday_rejection'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    high_open_diff = data['high'] - data['open']
    open_low_diff = data['open'] - data['low']
    data['efficiency_asymmetry'] = np.where(high_open_diff != 0, (data['close'] - data['open']) / high_open_diff, 0) - \
                                  np.where(open_low_diff != 0, (data['open'] - data['close']) / open_low_diff, 0)
    data['close_to_mid_range'] = data['close'] / ((data['high'] + data['low']) / 2) - 1
    
    # Persistence Integration
    # Price streak
    price_change_sign = np.sign(data['price_momentum'])
    price_streak = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if price_change_sign.iloc[i] == price_change_sign.iloc[i-1] and price_change_sign.iloc[i] != 0:
            price_streak.iloc[i] = price_streak.iloc[i-1] + 1
        else:
            price_streak.iloc[i] = 1 if price_change_sign.iloc[i] != 0 else 0
    data['price_streak'] = price_streak
    
    # Volume streak
    volume_change_sign = np.sign(data['volume_momentum'])
    volume_streak = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if volume_change_sign.iloc[i] == volume_change_sign.iloc[i-1] and volume_change_sign.iloc[i] != 0:
            volume_streak.iloc[i] = volume_streak.iloc[i-1] + 1
        else:
            volume_streak.iloc[i] = 1 if volume_change_sign.iloc[i] != 0 else 0
    data['volume_streak'] = volume_streak
    
    # Rejection persistence
    intraday_sign = np.sign(data['close'] - data['open'])
    rejection_persistence = (intraday_sign * data['intraday_rejection']).rolling(window=3, min_periods=1).sum()
    data['rejection_persistence'] = rejection_persistence
    
    # Asymmetry Components
    data['volatility_asymmetry'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) * \
                                  np.sign(data['open'] - data['close'].shift(1))
    
    # Volume median over 5 days
    volume_median_5d = data['volume'].rolling(window=5, min_periods=1).median()
    data['volume_pressure'] = data['volume'] / volume_median_5d * np.sign(data['close'] - data['open'])
    
    # Factor Synthesis
    data['momentum_rejection'] = data['price_momentum'] * data['intraday_rejection'] * data['close_to_mid_range']
    data['asymmetry_momentum'] = data['efficiency_asymmetry'] * data['volatility_asymmetry'] * data['price_momentum']
    data['persistence_pressure'] = (data['price_streak'] + data['volume_streak']) * data['volume_pressure'] * data['rejection_persistence']
    
    # Final factor combination
    factor = data['momentum_rejection'] + data['asymmetry_momentum'] + data['persistence_pressure']
    
    return factor
