import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Components
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['range_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1
    
    # Microstructure Signals
    data['intraday_rejection'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Efficiency asymmetry with safe division
    high_open_diff = data['high'] - data['open']
    open_low_diff = data['open'] - data['low']
    data['efficiency_asymmetry'] = np.where(high_open_diff != 0, (data['close'] - data['open']) / high_open_diff, 0) - \
                                  np.where(open_low_diff != 0, (data['open'] - data['close']) / open_low_diff, 0)
    
    data['close_to_mid_range'] = data['close'] / ((data['high'] + data['low']) / 2) - 1
    
    # Persistence Integration
    # Price streak
    price_changes = np.sign(data['price_momentum'])
    price_streak = pd.Series(index=data.index, dtype=float)
    current_streak = 0
    current_sign = 0
    for i in range(len(data)):
        if i == 0 or price_changes.iloc[i] == 0:
            current_streak = 0
            current_sign = 0
        elif price_changes.iloc[i] == current_sign:
            current_streak += 1
        else:
            current_streak = 1
            current_sign = price_changes.iloc[i]
        price_streak.iloc[i] = current_streak
    
    # Volume streak
    volume_changes = np.sign(data['volume_momentum'])
    volume_streak = pd.Series(index=data.index, dtype=float)
    current_streak = 0
    current_sign = 0
    for i in range(len(data)):
        if i == 0 or volume_changes.iloc[i] == 0:
            current_streak = 0
            current_sign = 0
        elif volume_changes.iloc[i] == current_sign:
            current_streak += 1
        else:
            current_streak = 1
            current_sign = volume_changes.iloc[i]
        volume_streak.iloc[i] = current_streak
    
    # Rejection persistence (3-day sum)
    sign_close_open = np.sign(data['close'] - data['open'])
    rejection_persistence = (sign_close_open * data['intraday_rejection']).rolling(window=3, min_periods=1).sum()
    
    # Asymmetry Components
    data['volatility_asymmetry'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) * \
                                  np.sign(data['open'] - data['close'].shift(1))
    
    volume_median_5d = data['volume'].rolling(window=5, min_periods=1).median()
    data['volume_pressure'] = (data['volume'] / volume_median_5d) * np.sign(data['close'] - data['open'])
    
    # Factor Synthesis
    momentum_rejection = data['price_momentum'] * data['intraday_rejection'] * data['close_to_mid_range']
    asymmetry_momentum = data['efficiency_asymmetry'] * data['volatility_asymmetry'] * data['price_momentum']
    persistence_pressure = (price_streak + volume_streak) * data['volume_pressure'] * rejection_persistence
    
    # Final factor combination
    factor = momentum_rejection + asymmetry_momentum + persistence_pressure
    
    return factor
