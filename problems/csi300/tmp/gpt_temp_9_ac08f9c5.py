import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Price-Volume Momentum
    # Short-term (3-day) Momentum
    price_return_3d = df['close'] / df['close'].shift(3) - 1
    volume_change_3d = df['volume'] / df['volume'].shift(3) - 1
    confirmed_momentum = price_return_3d * np.sign(volume_change_3d)
    
    # Medium-term (8-day) Momentum
    price_return_8d = df['close'] / df['close'].shift(8) - 1
    volume_trend_8d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 8:
            vol_trend_count = sum(df['volume'].iloc[i-j] > df['volume'].iloc[i-j-1] for j in range(8))
            volume_trend_8d.iloc[i] = vol_trend_count
    trend_quality = price_return_8d * volume_trend_8d
    
    # Combined Factor
    multi_timeframe_signal = confirmed_momentum + trend_quality
    recent_confirmation = multi_timeframe_signal * np.sign(df['close'] - df['close'].shift(1))
    
    # Range Efficiency with Volatility Context
    # Multi-period Efficiency
    price_change_3d = abs(df['close'] - df['close'].shift(3))
    range_sum_3d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            range_sum_3d.iloc[i] = sum(df['high'].iloc[i-j] - df['low'].iloc[i-j] for j in range(3))
    efficiency_3d = price_change_3d / range_sum_3d
    
    price_change_8d = abs(df['close'] - df['close'].shift(8))
    range_sum_8d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 8:
            range_sum_8d.iloc[i] = sum(df['high'].iloc[i-j] - df['low'].iloc[i-j] for j in range(8))
    efficiency_8d = price_change_8d / range_sum_8d
    
    efficiency_ratio = efficiency_3d / efficiency_8d
    
    # Volatility Context
    recent_volatility = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            recent_volatility.iloc[i] = np.mean([df['high'].iloc[i-j] - df['low'].iloc[i-j] for j in range(3)]) / df['close'].iloc[i]
    
    baseline_volatility = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 8:
            baseline_volatility.iloc[i] = np.mean([df['high'].iloc[i-j] - df['low'].iloc[i-j] for j in range(8)]) / df['close'].iloc[i]
    
    volatility_regime = recent_volatility / baseline_volatility
    
    # Final Factor
    efficiency_context = efficiency_ratio * volatility_regime
    volume_validation = efficiency_context * np.sign(df['volume'] - df['volume'].shift(3))
    
    # Breakout Patterns with Multi-dimensional Confirmation
    # Breakout Detection
    breakout_3d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            breakout_3d.iloc[i] = float(df['close'].iloc[i] > max(df['high'].iloc[i-2:i]))
    
    breakout_8d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 8:
            breakout_8d.iloc[i] = float(df['close'].iloc[i] > max(df['high'].iloc[i-7:i]))
    
    breakout_strength = breakout_3d + breakout_8d
    
    # Confirmation Signals
    volume_surge = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            volume_surge.iloc[i] = float(df['volume'].iloc[i] > np.mean(df['volume'].iloc[i-2:i]))
    
    amount_intensity = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            amount_intensity.iloc[i] = float(df['amount'].iloc[i] > np.mean(df['amount'].iloc[i-2:i]))
    
    confirmation_score = volume_surge + amount_intensity
    
    # Final Factor
    confirmed_breakout = breakout_strength * confirmation_score
    momentum_enhancement = confirmed_breakout * (df['close'] / df['close'].shift(1) - 1)
    
    # Amount Flow Direction with Persistence
    # Short-term Flow (3-day)
    directional_amount_3d = pd.Series(index=df.index, dtype=float)
    flow_consistency_3d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            dir_amount = sum(df['amount'].iloc[i-j] * np.sign(df['close'].iloc[i-j] - df['close'].iloc[i-j-1]) for j in range(3))
            directional_amount_3d.iloc[i] = dir_amount
            
            consistency_count = sum(np.sign(df['amount'].iloc[i-j] * (df['close'].iloc[i-j] - df['close'].iloc[i-j-1])) > 0 for j in range(3))
            flow_consistency_3d.iloc[i] = consistency_count
    
    flow_momentum_3d = directional_amount_3d * flow_consistency_3d
    
    # Medium-term Context (8-day)
    cumulative_flow_8d = pd.Series(index=df.index, dtype=float)
    flow_persistence_8d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 8:
            cum_flow = sum(df['amount'].iloc[i-j] * np.sign(df['close'].iloc[i-j] - df['close'].iloc[i-j-1]) for j in range(8))
            cumulative_flow_8d.iloc[i] = cum_flow
            
            persistence_count = sum(np.sign(df['amount'].iloc[i-j] * (df['close'].iloc[i-j] - df['close'].iloc[i-j-1])) > 0 for j in range(8))
            flow_persistence_8d.iloc[i] = persistence_count
    
    flow_quality_8d = cumulative_flow_8d * flow_persistence_8d
    
    # Combined Factor
    multi_timeframe_flow = flow_momentum_3d * np.sign(flow_quality_8d)
    volume_alignment = multi_timeframe_flow * np.sign(df['volume'] - df['volume'].shift(3))
    
    # Overnight-Intraday Momentum Patterns
    # Gap Analysis
    overnight_return = df['open'] / df['close'].shift(1) - 1
    intraday_return = df['close'] / df['open'] - 1
    gap_continuation = overnight_return * intraday_return
    
    # Volume Validation
    overnight_volume = df['volume'] / df['volume'].shift(1)
    intraday_volume_pattern = (df['volume'] - df['volume'].shift(1)) / (df['high'] - df['low'])
    volume_confirmation = gap_continuation * overnight_volume * intraday_volume_pattern
    
    # Final Factor
    multi_day_context = volume_confirmation * np.sign(df['close'] - df['close'].shift(3))
    amount_enhancement = multi_day_context * (df['amount'] / df['amount'].shift(1) - 1)
    
    # Combine all factors with equal weighting
    final_factor = (recent_confirmation + volume_validation + momentum_enhancement + 
                   volume_alignment + amount_enhancement) / 5
    
    return final_factor
