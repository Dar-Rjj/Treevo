import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['daily_midpoint'] = (data['high'] + data['low']) / 2
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Directional Movement
    data['plus_dm'] = data['high'] - data['high'].shift(1)
    data['minus_dm'] = data['low'].shift(1) - data['low']
    data['plus_dm'] = np.where(data['plus_dm'] > 0, data['plus_dm'], 0)
    data['minus_dm'] = np.where(data['minus_dm'] > 0, data['minus_dm'], 0)
    
    # ADX components (14-day)
    period = 14
    data['tr_smooth'] = data['true_range'].ewm(span=period, adjust=False).mean()
    data['plus_dm_smooth'] = data['plus_dm'].ewm(span=period, adjust=False).mean()
    data['minus_dm_smooth'] = data['minus_dm'].ewm(span=period, adjust=False).mean()
    
    data['plus_di'] = 100 * data['plus_dm_smooth'] / data['tr_smooth']
    data['minus_di'] = 100 * data['minus_dm_smooth'] / data['tr_smooth']
    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    data['adx'] = data['dx'].ewm(span=period, adjust=False).mean()
    
    # Volume Weighted Average Price
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    data['vwap_deviation'] = (data['vwap'] - data['daily_midpoint']) / data['daily_midpoint']
    
    # Volume acceleration
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Price-volume correlation (5-day)
    data['price_change'] = data['close'].pct_change()
    data['price_volume_corr'] = data['price_change'].rolling(window=5).corr(data['volume'].pct_change())
    
    # Volume confirmation score
    data['volume_confirmation'] = (
        np.sign(data['vwap_deviation']) * data['adx'] / 100 +
        data['volume_acceleration'].fillna(0) +
        data['price_volume_corr'].fillna(0)
    )
    
    # Trend persistence metrics
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['consecutive_days'] = 0
    for i in range(1, len(data)):
        if data['price_direction'].iloc[i] == data['price_direction'].iloc[i-1]:
            data.loc[data.index[i], 'consecutive_days'] = data['consecutive_days'].iloc[i-1] + 1
    
    # ADX slope (3-day)
    data['adx_slope'] = data['adx'].diff(3) / 3
    
    # Trend persistence score
    data['trend_persistence'] = (
        data['consecutive_days'] * 0.1 +
        data['adx_slope'].fillna(0) * 10
    )
    
    # Overnight gap analysis
    data['overnight_gap'] = data['open'] / data['prev_close'] - 1
    
    # Momentum crossover (5-day vs 10-day)
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['momentum_crossover'] = np.where(data['sma_5'] > data['sma_10'], 1, -1)
    
    # Final signal integration
    data['trend_strength_volume'] = data['adx'] * data['volume_confirmation']
    
    data['final_signal'] = (
        data['trend_strength_volume'] * 0.4 +
        data['trend_persistence'] * 0.3 +
        data['overnight_gap'] * 100 * 0.2 +
        data['momentum_crossover'] * 0.1
    )
    
    return data['final_signal']
