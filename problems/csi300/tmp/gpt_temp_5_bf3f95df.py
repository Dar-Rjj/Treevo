import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required rolling windows
    df['high_roll_4'] = df['high'].rolling(window=5, min_periods=5).max()
    df['low_roll_4'] = df['low'].rolling(window=5, min_periods=5).min()
    df['high_roll_2'] = df['high'].rolling(window=3, min_periods=3).max()
    df['low_roll_2'] = df['low'].rolling(window=3, min_periods=3).min()
    
    # Fractal Efficiency Dynamics
    df['micro_fractal'] = np.abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 0.0001)
    df['macro_fractal'] = np.abs(df['close'] - df['close'].shift(5)) / (df['high_roll_4'] - df['low_roll_4'] + 0.0001)
    df['fractal_efficiency_ratio'] = df['micro_fractal'] / (df['macro_fractal'] + 0.0001)
    
    # Asymmetric Pressure Dynamics
    df['immediate_pressure'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 0.0001)
    df['short_term_pressure'] = (df['close'] - df['close'].shift(3)) / (df['high_roll_2'] - df['low_roll_2'] + 0.0001)
    
    pressure_convergence = np.sign(df['immediate_pressure']) * np.sign(df['short_term_pressure']) * (1 / (1 + np.abs(df['immediate_pressure'] - df['short_term_pressure'])))
    
    df['upper_pressure'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 0.0001)
    df['lower_pressure'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['pressure_asymmetry'] = df['upper_pressure'] - df['lower_pressure']
    
    # Volume Microstructure
    df['volume_concentration'] = df['volume'] / (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2) + 0.0001)
    
    volume_volatility_entropy = df['volume'] / (df['volume'] - df['volume'].shift(1) + 0.0001) * (df['volume'] / (df['volume'].shift(1) + 0.0001))
    df['volume_pressure_integration'] = volume_volatility_entropy * df['pressure_asymmetry']
    
    # Regime Detection
    volatility_regime_filter = ((df['high'] - df['low']) / (np.abs(df['close'] - df['open']) + 0.0001)) > 1.5
    volume_regime_filter = (df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 0.0001) > 0
    
    price_breakout = (df['close'] > df['high'].shift(1)) & (df['volume'] > df['volume'].shift(1))
    
    volume_ma_3 = (df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3)) / 3
    volume_breakout = df['volume'] > (1.5 * volume_ma_3)
    
    regime_multiplier = 1.0 + 0.4 * volatility_regime_filter.astype(float) + 0.3 * volume_regime_filter.astype(float) + 0.2 * (price_breakout.astype(float) + volume_breakout.astype(float))
    
    # Amount-Volume Dynamics
    df['flow_momentum'] = (df['close'] - df['open']) * df['volume'] / (df['amount'] + 0.0001)
    df['volume_acceleration'] = (df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 0.0001)
    
    # Volume Trend Consistency (5-day correlation)
    volume_trend_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 5:
            window_data = df.iloc[i-4:i+1]
            if len(window_data) == 5:
                volume_data = window_data['volume'].values
                price_change_data = np.abs(window_data['close'].values - window_data['close'].shift(1).values)[1:]
                
                if len(price_change_data) == 4 and not (np.std(volume_data) == 0 or np.std(price_change_data) == 0):
                    correlation = np.corrcoef(volume_data[1:], price_change_data)[0, 1]
                    if not np.isnan(correlation):
                        volume_trend_consistency.iloc[i] = correlation
    
    # Alpha Synthesis
    core_components = [
        df['fractal_efficiency_ratio'] * pressure_convergence,
        df['fractal_efficiency_ratio'] * df['volume_pressure_integration'],
        df['flow_momentum'] * df['volume_concentration']
    ]
    
    base_alpha = sum(core_components) * df['pressure_asymmetry']
    
    # Final Alpha
    final_alpha = base_alpha * regime_multiplier * df['volume_acceleration'] * volume_trend_consistency
    
    # Fill NaN values with 0
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
