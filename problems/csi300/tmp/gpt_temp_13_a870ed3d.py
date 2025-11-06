import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Fractal Momentum Acceleration
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['momentum_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    df['short_acceleration'] = (df['momentum_3d'] - df['momentum_5d']) / 2
    df['medium_acceleration'] = (df['momentum_5d'] - df['momentum_10d']) / 5
    df['acceleration_strength'] = df['momentum_5d'] * (df['short_acceleration'] + df['medium_acceleration']) / 2
    
    # Intraday Pressure Dynamics
    df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).replace([np.inf, -np.inf], 0).fillna(0) * df['volume']
    df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'])).replace([np.inf, -np.inf], 0).fillna(0) * df['volume']
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    
    df['price_volume_alignment'] = np.sign(df['momentum_3d']) * np.sign(df['volume'] - df['volume'].shift(3)) * np.abs(df['momentum_3d'] * (df['volume'] - df['volume'].shift(3)))
    df['pressure_signal'] = df['price_volume_alignment'] * np.sign(df['net_pressure'])
    
    # Volatility Breakout System
    df['rolling_max_high'] = df['high'].rolling(window=5, min_periods=1).max()
    df['rolling_min_low'] = df['low'].rolling(window=5, min_periods=1).min()
    
    df['upper_breakout'] = (df['close'] - df['rolling_max_high']) * np.abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-10)
    df['lower_breakout'] = (df['close'] - df['rolling_min_low']) * (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-10)
    df['breakout_score'] = df['upper_breakout'] + df['lower_breakout']
    
    df['daily_range'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2).replace(0, 1e-10)
    df['volatility_breakout'] = df['breakout_score'] / df['daily_range'].replace(0, 1e-10)
    
    # Flow Persistence Patterns
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_change_sign'] = np.sign(df['price_change'])
    
    consecutive_direction = []
    for i in range(len(df)):
        if i < 5:
            consecutive_direction.append(0)
        else:
            window = df['price_change_sign'].iloc[i-4:i+1]
            if len(window) == 5 and all(sign == window.iloc[-1] for sign in window):
                consecutive_direction.append(5)
            else:
                count = 1
                for j in range(len(window)-2, -1, -1):
                    if window.iloc[j] == window.iloc[-1]:
                        count += 1
                    else:
                        break
                consecutive_direction.append(count)
    
    df['consecutive_direction'] = consecutive_direction
    
    volume_correlation = []
    for i in range(len(df)):
        if i < 5:
            volume_correlation.append(0)
        else:
            vol_window = df['volume'].iloc[i-4:i+1]
            if vol_window.std() == 0:
                volume_correlation.append(0)
            else:
                corr = np.corrcoef(range(5), vol_window.values)[0, 1]
                volume_correlation.append(corr if not np.isnan(corr) else 0)
    
    df['volume_correlation'] = volume_correlation
    
    df['flow_consistency'] = df['consecutive_direction'] * df['volume_correlation']
    df['persistence_momentum'] = df['flow_consistency'] * (1 - np.abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, 1e-10))
    
    # Composite Alpha Generation
    df['core_factor'] = df['acceleration_strength'] * df['pressure_signal'] * df['volatility_breakout']
    df['flow_enhanced'] = df['core_factor'] * df['persistence_momentum']
    
    df['volume_avg_10d'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['final_alpha'] = df['flow_enhanced'] * (df['volume'] / df['volume_avg_10d'].replace(0, 1e-10))
    
    return df['final_alpha']
