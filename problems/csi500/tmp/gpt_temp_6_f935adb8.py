import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price components
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    # Gap Absorption Dynamics
    df['gap_size'] = (df['open'] - df['prev_close']) / (df['prev_high'] - df['prev_low'])
    df['gap_persistence'] = (df['close'] - df['prev_close']) / (abs(df['open'] - df['prev_close']) + 1e-8)
    df['intraday_absorption'] = (df['close'] - df['open']) / (abs(df['open'] - df['prev_close']) + 1e-8)
    df['multi_day_validation'] = (df['close'] - df['close'].shift(3)) / (abs(df['open'] - df['prev_close']) + 1e-8)
    
    # Volume-Pressure Alignment
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_momentum'] = df['volume'] / (df['volume_ma_5'] + 1e-8)
    
    # Volume trend persistence using rolling correlation
    def volume_correlation(series):
        if len(series) < 10:
            return np.nan
        recent = series[-5:]
        previous = series[-10:-5]
        return np.corrcoef(recent, previous)[0,1] if len(recent) == len(previous) == 5 else np.nan
    
    df['volume_trend_persistence'] = df['volume'].rolling(window=10).apply(volume_correlation, raw=False)
    
    # True Range calculation
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Volatility Regime Context
    df['true_range_ma_10'] = df['true_range'].rolling(window=10).mean()
    df['current_volatility'] = df['true_range'] / (df['true_range_ma_10'] + 1e-8)
    df['volatility_trend'] = df['true_range'] / (df['true_range'].shift(5) + 1e-8)
    
    df['price_efficiency'] = abs(df['close'] - df['open']) / (df['true_range'] + 1e-8)
    df['efficiency_trend'] = df['price_efficiency'] - df['price_efficiency'].shift(3)
    
    # Pressure Accumulation
    df['intraday_pressure'] = (df['close'] - df['open']) * df['volume']
    df['pressure_accumulation_3d'] = df['intraday_pressure'].rolling(window=3).sum()
    df['pressure_ma_5'] = df['pressure_accumulation_3d'].rolling(window=5).mean()
    
    # Breakout Confirmation Signals
    df['pressure_threshold'] = df['pressure_accumulation_3d'] > df['pressure_ma_5']
    df['volume_expansion'] = df['volume'] > df['volume_ma_5'] * 1.2
    
    # Exhaustion Detection
    df['volume_derivative'] = df['volume'].diff(3)
    df['price_rejection'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)) > 0.7
    
    # Primary Momentum Component
    gap_absorption_strength = (df['intraday_absorption'] * df['gap_persistence']).fillna(0)
    volume_momentum_alignment = df['volume_momentum'] * df['volume_trend_persistence'].fillna(0)
    volatility_adjustment = 1 / (df['current_volatility'] + 1e-8)
    
    primary_momentum = gap_absorption_strength * volume_momentum_alignment * volatility_adjustment
    
    # Secondary Confirmation
    pressure_confirmation = df['pressure_accumulation_3d'] * df['pressure_threshold'].astype(float)
    breakout_confirmation = pressure_confirmation * df['volume_expansion'].astype(float)
    
    exhaustion_risk = 1 / (1 + abs(df['volume_derivative']) + df['price_rejection'].astype(float))
    
    secondary_confirmation = breakout_confirmation * exhaustion_risk
    
    # Final Alpha Construction
    efficiency_weight = 1 + df['efficiency_trend'].fillna(0)
    
    alpha = primary_momentum * secondary_confirmation * efficiency_weight
    
    return alpha
