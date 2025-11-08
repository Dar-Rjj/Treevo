import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Divergence
    # Short-term Momentum (3-day)
    mom_3d = df['close'] / df['close'].shift(3) - 1
    intraday_conf_3d = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Medium-term Momentum (5-day)
    mom_5d = df['close'] / df['close'].shift(5) - 1
    range_eff_5d = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Long-term Momentum (10-day)
    mom_10d = df['close'] / df['close'].shift(10) - 1
    trend_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        window = df['close'].iloc[i-9:i+1]
        trend_persistence.iloc[i] = (window > window.shift(1)).sum()
    
    # Combined Momentum Signal
    momentum_alignment = np.sign(mom_3d * mom_5d * mom_10d) * (np.abs(mom_3d * mom_5d * mom_10d))**(1/3)
    intraday_strength = (intraday_conf_3d + range_eff_5d) / 2
    volatility_scaling = (df['high'] - df['low']) / df['close']
    momentum_signal = momentum_alignment * intraday_strength / volatility_scaling
    
    # Volume-Price Convergence Factor
    # Price Strength Components
    opening_strength = (df['close'] - df['open']) / (df['high'] - df['low'])
    closing_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    daily_efficiency = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Volume Confirmation
    vol_acceleration = df['volume'] / df['volume'].shift(1)
    vol_trend = df['volume'] / ((df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3)) / 3)
    vol_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window = df['volume'].iloc[i-4:i+1]
        vol_persistence.iloc[i] = (window > window.shift(1)).sum()
    
    # Combined Volume-Price Signal
    core_price_strength = (opening_strength + closing_position + daily_efficiency) / 3
    volume_multiplier = np.sign(vol_acceleration * vol_trend) * (np.abs(vol_acceleration * vol_trend))**(1/2)
    volume_price_signal = core_price_strength * volume_multiplier
    
    # Intraday Session Alignment
    # Morning Session Analysis
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    morning_range = (df['high'] - df['open']) / df['open']
    morning_support = (df['open'] - df['low']) / df['open']
    
    # Afternoon Session Analysis
    afternoon_momentum = (df['close'] - df['high']) / df['high']
    closing_strength = (df['close'] - df['low']) / df['low']
    session_consistency = (df['close'] - df['open']) / df['open']
    
    # Combined Session Signal
    session_alignment = (morning_range - morning_support) * (afternoon_momentum - closing_strength)
    intraday_signal = session_alignment * opening_gap * session_consistency
    
    # Multi-Timeframe Signal Integration
    # Short-term Convergence (1-3 days)
    price_mom_st = df['close'] / df['close'].shift(2) - 1
    volume_mom_st = df['volume'] / df['volume'].shift(2)
    range_util_st = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Medium-term Convergence (5-10 days)
    price_trend_mt = df['close'] / df['close'].shift(7) - 1
    volume_trend_mt = df['volume'] / ((df['volume'].shift(4) + df['volume'].shift(5) + df['volume'].shift(6)) / 3)
    vol_persistence_mt = (df['high'] - df['low']) / ((df['high'].shift(4) - df['low'].shift(4) + 
                                                     df['high'].shift(5) - df['low'].shift(5) + 
                                                     df['high'].shift(6) - df['low'].shift(6)) / 3)
    
    # Combined Multi-Timeframe Signal
    timeframe_alignment = price_mom_st * price_trend_mt
    volume_convergence = volume_mom_st * volume_trend_mt
    efficiency_adj = range_util_st / vol_persistence_mt
    multi_timeframe_signal = timeframe_alignment * volume_convergence * efficiency_adj
    
    # Final Alpha Factor Combination
    alpha_factor = (momentum_signal.rank(pct=True) + 
                   volume_price_signal.rank(pct=True) + 
                   intraday_signal.rank(pct=True) + 
                   multi_timeframe_signal.rank(pct=True)) / 4
    
    return alpha_factor
