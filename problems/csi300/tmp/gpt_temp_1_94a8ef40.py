import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Reversal-Efficiency Alpha with Regime-Switching Memory
    """
    data = df.copy()
    
    # Helper functions
    def safe_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    
    # Multi-Timeframe Asymmetric Reversal Detection
    # Ultra-Short Asymmetric Reversal (2-day)
    gap_reversal = np.sign(data['open'] - data['close'].shift(1)) * np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_pressure = safe_divide((data['high'] - data['open']) - (data['open'] - data['low']), data['open'])
    session_efficiency = safe_divide(data['close'] - data['open'], data['high'] - data['low']) * np.sign(data['close'] - data['open'])
    ultra_short_reversal = 0.4 * gap_reversal + 0.3 * intraday_pressure + 0.3 * session_efficiency
    
    # Short-Term Asymmetric Reversal (5-day)
    directional_oscillation = np.sign(data['close'] - data['close'].shift(3)) * safe_divide(
        np.abs(data['close'] - data['close'].shift(3)), 
        np.abs(data['close'].shift(1) - data['close'].shift(2))
    )
    range_compression = safe_divide(data['high'] - data['low'], data['high'].shift(3) - data['low'].shift(3)) * np.sign(data['close'] - data['close'].shift(3))
    volatility_regime = data['high'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    range_compression_asym = range_compression * volatility_regime
    short_term_reversal = 0.4 * directional_oscillation + 0.4 * range_compression_asym + 0.2 * volatility_regime
    
    # Medium-Term Asymmetric Reversal (20-day)
    key_level_pressure = np.sign(data['close'] - data['close'].shift(1)) * safe_divide(data['high'] - data['low'], data['close'])
    buy_pressure = safe_divide(data['close'] - data['low'], data['high'] - data['low'])
    sell_pressure = safe_divide(data['high'] - data['close'], data['high'] - data['low'])
    support_resistance_asym = buy_pressure - sell_pressure
    structural_pressure = np.abs(
        safe_divide(data['high'] - data['open'], data['close'] - data['low']) - 
        safe_divide(data['close'] - data['low'], data['high'] - data['open'])
    ) * np.sign(data['close'] - data['open'])
    medium_term_reversal = 0.4 * key_level_pressure + 0.3 * support_resistance_asym + 0.3 * structural_pressure
    
    # Multi-timeframe asymmetric reversal composite
    multi_timeframe_reversal = 0.3 * ultra_short_reversal + 0.35 * short_term_reversal + 0.35 * medium_term_reversal
    
    # Efficiency-Momentum with Memory Effects
    # Multi-Timeframe Efficiency Acceleration
    two_day_efficiency = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    five_day_efficiency = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    twenty_day_efficiency = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    recent_memory = data['close'].rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    intermediate_memory = data['close'].rolling(window=10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    ultra_short_accel = (two_day_efficiency - five_day_efficiency) * recent_memory
    short_term_accel = (five_day_efficiency - twenty_day_efficiency) * intermediate_memory
    regime_accel = (ultra_short_accel + short_term_accel) * volatility_regime
    
    # Momentum Exhaustion with Memory
    morning_afternoon_momentum = ((data['high'] - data['open']) - (data['close'] - data['low'])) * recent_memory
    trend_regime = data['close'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.mean() else -1)
    momentum_sustainability = session_efficiency * trend_regime
    
    fade_strength = np.abs(session_efficiency.rolling(window=5).mean() - session_efficiency)
    persistent_memory = data['close'].rolling(window=20).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    fade_accumulation = fade_strength.rolling(window=5).sum() * persistent_memory
    
    # Volume-Order Flow Synchronization
    # Asymmetric Volume Dynamics
    directional_volume_accel = safe_divide(data['volume'], data['volume'].shift(1)) - 1 * np.sign(data['close'] - data['open'])
    volume_var_3d = data['volume'].rolling(window=3).var()
    volume_var_10d = data['volume'].rolling(window=10).var()
    volume_clustering_asym = safe_divide(volume_var_3d, volume_var_10d) * (buy_pressure - sell_pressure)
    
    volume_median_20d = data['volume'].rolling(window=20).median()
    extreme_volume_pressure = (data['volume'] > 2 * volume_median_20d) * (buy_pressure - sell_pressure)
    
    # Order Flow Pressure Imbalance
    directional_amount_pressure = data['amount'] * (buy_pressure - sell_pressure)
    
    def calc_cumulative_pressure(window):
        pressure_sum = 0
        amount_sum = 0
        for i in range(window):
            asymmetry = buy_pressure.shift(i) - sell_pressure.shift(i)
            pressure_sum += data['amount'].shift(i) * asymmetry
            amount_sum += np.abs(data['amount'].shift(i))
        return safe_divide(pressure_sum, amount_sum)
    
    pressure_imbalance_ratio = calc_cumulative_pressure(3)
    
    # Volatility-Pressure Context
    intraday_pressure_vol = safe_divide(data['high'] - data['low'], data['close'].shift(1)) * (buy_pressure - sell_pressure)
    vol_persistence_pressure = safe_divide(data['high'] - data['low'], data['high'].shift(1) - data['low'].shift(1)) * pressure_imbalance_ratio
    efficiency_pressure_alignment = session_efficiency * vol_persistence_pressure
    
    # Regime-Switching Position Dynamics
    range_position = safe_divide(data['close'] - data['low'], data['high'] - data['low']) * trend_regime
    position_deviation_asym = np.abs(range_position - 0.5) * 2 * (buy_pressure - sell_pressure)
    
    session_dominance = safe_divide(
        np.maximum(data['high'] - data['open'], data['open'] - data['low']),
        data['high'] - data['low']
    ) * (data['volume'] / data['volume'].rolling(window=20).mean())
    
    # Memory-Enhanced Breakout Assessment
    key_level_break = ((data['close'] > data['high'].shift(1)) | (data['close'] < data['low'].shift(1))) * recent_memory
    breakout_efficiency = safe_divide(np.abs(data['close'] - data['close'].shift(1)), data['high'] - data['low']) * pressure_imbalance_ratio
    false_breakout_prob = 1 - breakout_efficiency * volatility_regime
    
    # Cross-Regime Position Validation
    trend_vol_alignment = trend_regime * volatility_regime * (buy_pressure - sell_pressure)
    liquidity_pressure_conf = (data['volume'] / data['volume'].rolling(window=20).mean()) * pressure_imbalance_ratio * volume_clustering_asym
    memory_consistency = persistent_memory * intermediate_memory * recent_memory
    
    # Composite Asymmetric Alpha with Memory
    # Core Asymmetric Components
    core_asymmetric = multi_timeframe_reversal
    efficiency_accel_memory = regime_accel * 0.25
    momentum_exhaustion = fade_accumulation * 0.2
    
    # Volume-Order Flow Confirmation
    volume_pressure_align = directional_volume_accel * np.sign(regime_accel) * 0.2
    order_flow_sync = pressure_imbalance_ratio * volume_clustering_asym * 0.2
    volatility_pressure_adj = efficiency_pressure_alignment * 0.15
    
    # Regime-Memory Adjustment
    regime_conditional = (trend_vol_alignment + liquidity_pressure_conf) * 0.25
    memory_weighted_pos = memory_consistency * position_deviation_asym * 0.15
    breakout_validity = false_breakout_prob * -0.1
    
    # Final Alpha Signal
    alpha_signal = (
        core_asymmetric +
        efficiency_accel_memory +
        momentum_exhaustion +
        volume_pressure_align +
        order_flow_sync +
        volatility_pressure_adj +
        regime_conditional +
        memory_weighted_pos +
        breakout_validity
    )
    
    # Normalize and return
    alpha_normalized = (alpha_signal - alpha_signal.rolling(window=20).mean()) / alpha_signal.rolling(window=20).std()
    return alpha_normalized.fillna(0)
