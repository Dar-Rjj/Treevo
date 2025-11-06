import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Multi-Timeframe Volatility Assessment
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    vol_ratio = vol_5d / vol_20d
    
    # Volatility Regime Classification
    vol_5d_75th = vol_5d.rolling(window=60).quantile(0.75)
    vol_5d_25th = vol_5d.rolling(window=60).quantile(0.25)
    vol_20d_75th = vol_20d.rolling(window=60).quantile(0.75)
    vol_20d_25th = vol_20d.rolling(window=60).quantile(0.25)
    
    high_vol_regime = (vol_5d > vol_5d_75th) & (vol_20d > vol_20d_75th)
    low_vol_regime = (vol_5d < vol_5d_25th) & (vol_20d < vol_20d_25th)
    transition_regime = ~high_vol_regime & ~low_vol_regime
    
    # Volume-Pressure Momentum Calculation
    buy_pressure = (data['high'] - data['close']) * data['volume']
    sell_pressure = (data['close'] - data['low']) * data['volume']
    net_pressure = buy_pressure - sell_pressure
    
    # 3-day accumulated net pressure
    acc_net_pressure = net_pressure.rolling(window=3).sum()
    
    # Volume-Weighted Momentum with exponential smoothing
    volume_weighted_momentum = acc_net_pressure.ewm(span=5).mean()
    
    # Scale by daily volume percentile rank
    volume_rank = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_weighted_momentum = volume_weighted_momentum * volume_rank
    
    # Regime-Adaptive Efficiency Measurement
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Price Movement Efficiency
    price_efficiency = abs(data['close'] - data['close'].shift(1)) / true_range
    
    # Volatility-Adjusted Swing Ratio
    atr_5d = true_range.rolling(window=5).mean()
    swing_ratio = (data['high'] - data['low']) / atr_5d
    
    # Combined efficiency measure
    efficiency_measure = price_efficiency * swing_ratio
    
    # Regime-specific adjustments
    regime_adjusted_efficiency = efficiency_measure.copy()
    regime_adjusted_efficiency[high_vol_regime] = efficiency_measure[high_vol_regime] * 1.5
    regime_adjusted_efficiency[low_vol_regime] = efficiency_measure[low_vol_regime] * 0.8
    regime_adjusted_efficiency[transition_regime] = efficiency_measure[transition_regime] * 1.0
    
    # Breakout Signal Generation - Core Momentum Factor
    core_momentum = volume_weighted_momentum * regime_adjusted_efficiency
    
    # Volume Confirmation
    volume_median_10d = data['volume'].rolling(window=10).median()
    volume_ratio = data['volume'] / volume_median_10d
    volume_confirmation = volume_ratio > 1.5
    
    # Count consecutive days with confirming volume
    volume_confirmation_count = volume_confirmation.astype(int)
    for i in range(1, len(volume_confirmation_count)):
        if volume_confirmation.iloc[i]:
            volume_confirmation_count.iloc[i] = volume_confirmation_count.iloc[i-1] + 1
        else:
            volume_confirmation_count.iloc[i] = 0
    
    # Regime-Adaptive Volume Confirmation Score
    volume_score = volume_confirmation_count.copy()
    volume_score[high_vol_regime] = volume_confirmation_count[high_vol_regime] * 1.0
    volume_score[low_vol_regime] = (volume_confirmation_count[low_vol_regime] >= 2).astype(float)
    volume_score[transition_regime] = volume_confirmation_count[transition_regime] * 0.7
    
    # Final Alpha Factor Construction
    # Regime-based weighting
    momentum_component = core_momentum.copy()
    volume_component = volume_score.copy()
    
    # Normalize components
    momentum_z = (momentum_component - momentum_component.rolling(window=20).mean()) / momentum_component.rolling(window=20).std()
    volume_z = (volume_component - volume_component.rolling(window=20).mean()) / volume_component.rolling(window=20).std()
    
    # Apply regime weights
    final_factor = pd.Series(index=data.index, dtype=float)
    final_factor[high_vol_regime] = momentum_z[high_vol_regime] * 0.6 + volume_z[high_vol_regime] * 0.4
    final_factor[low_vol_regime] = momentum_z[low_vol_regime] * 0.4 + volume_z[low_vol_regime] * 0.6
    final_factor[transition_regime] = momentum_z[transition_regime] * 0.5 + volume_z[transition_regime] * 0.5
    
    # Acceleration Enhancement
    momentum_3d = volume_weighted_momentum.rolling(window=3).mean()
    momentum_1d = volume_weighted_momentum
    momentum_acceleration = (momentum_3d - momentum_1d)
    
    # Scale by 5-day price range
    price_range_5d = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    acceleration_enhancement = momentum_acceleration / price_range_5d
    
    # Apply acceleration enhancement during acceleration phases
    acceleration_phase = momentum_acceleration > momentum_acceleration.rolling(window=5).mean()
    final_factor[acceleration_phase] = final_factor[acceleration_phase] * (1 + abs(acceleration_enhancement[acceleration_phase]))
    
    # Fill NaN values
    final_factor = final_factor.fillna(0)
    
    return final_factor
