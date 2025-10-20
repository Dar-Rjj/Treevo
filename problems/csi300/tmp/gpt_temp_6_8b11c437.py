import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Synchronization with Efficiency Persistence
    """
    data = df.copy()
    
    # Momentum Persistence Efficiency
    # Multi-Timeframe Momentum Consistency
    # Short-Term Momentum Persistence
    data['momentum_short_persistence'] = data['close'].rolling(window=5).apply(
        lambda x: np.sum(x > x.shift(1)) / 4 if len(x) == 5 else np.nan, raw=False
    )
    
    # Medium-Term Momentum Strength
    data['momentum_medium_strength'] = (data['close'] / data['close'].shift(10)) - (data['close'].shift(10) / data['close'].shift(20))
    
    # Momentum Acceleration Ratio
    mom_accel_ratio = (data['close'] / data['close'].shift(5) - 1) / (data['close'] / data['close'].shift(20) - 1)
    data['momentum_accel_ratio'] = mom_accel_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Momentum Efficiency Persistence
    data['momentum_efficiency_persistence'] = data['momentum_accel_ratio'].diff(3).fillna(0)
    
    # Volume-Price Synchronization Analysis
    # Volume Convergence Assessment
    # Volume-Price Correlation (5-day rolling)
    data['volume_price_corr'] = data['volume'].rolling(window=5).corr(data['close'])
    
    # Volume Stability
    data['volume_stability'] = data['volume'] / data['volume'].rolling(window=20).mean().shift(1)
    
    # Volume Acceleration Analysis
    vol_accel_short = data['volume'] / data['volume'].shift(2) - 1
    vol_accel_medium = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration_ratio'] = vol_accel_short / vol_accel_medium.replace(0, np.nan)
    data['volume_acceleration_ratio'] = data['volume_acceleration_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Weighted Efficiency
    # VWAP calculation
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    data['vwap_deviation'] = (data['close'] - data['vwap']) / data['close']
    
    # 3-day cumulative pressure
    data['daily_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    data['cumulative_pressure'] = data['daily_efficiency'].rolling(window=3).sum()
    
    # Dynamic Range Efficiency Integration
    # Current Range Efficiency Assessment
    data['intraday_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    
    # Range expansion context
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=5).mean().shift(1)
    
    # Range efficiency trend
    data['range_efficiency_trend'] = data['intraday_efficiency'].diff(3)
    
    # Synchronization Strength Quantification
    # Momentum-Volume Synchronization Score
    momentum_persistence = data['momentum_short_persistence'].fillna(0)
    volume_corr = data['volume_price_corr'].fillna(0)
    volume_stab = data['volume_stability'].fillna(1)
    
    data['momentum_volume_sync'] = (momentum_persistence * volume_corr * volume_stab)
    
    # Multi-Timeframe Efficiency Alignment
    range_efficiency = data['range_efficiency_trend'].fillna(0)
    vwap_efficiency = data['vwap_deviation'].abs().fillna(0)
    
    data['multi_timeframe_alignment'] = (
        momentum_persistence * range_efficiency + 
        data['momentum_medium_strength'].fillna(0) * (1 - vwap_efficiency)
    )
    
    # Pressure Accumulation Validation
    cumulative_pressure = data['cumulative_pressure'].fillna(0)
    volume_confirmation = (volume_corr.abs() * volume_stab).fillna(0)
    
    data['pressure_volume_sync'] = cumulative_pressure * volume_confirmation
    
    # Composite Alpha Generation
    # Signal Integration Framework
    momentum_efficiency = data['momentum_efficiency_persistence'].fillna(0)
    volume_sync = data['momentum_volume_sync'].fillna(0)
    range_efficiency_weight = (1 + data['range_expansion'].fillna(1)).clip(0.5, 2)
    sync_strength = (data['multi_timeframe_alignment'].fillna(0) + data['pressure_volume_sync'].fillna(0)) / 2
    
    # Efficiency Persistence Validation
    momentum_persistence_eff = data['momentum_accel_ratio'].rolling(window=3).std().fillna(1)
    range_persistence = data['intraday_efficiency'].rolling(window=3).std().fillna(1)
    
    efficiency_persistence = (1 / (momentum_persistence_eff * range_persistence)).replace([np.inf, -np.inf], 1)
    
    # Final Alpha Factor
    alpha = (
        momentum_efficiency * 
        volume_sync * 
        range_efficiency_weight * 
        sync_strength * 
        efficiency_persistence
    )
    
    return alpha
