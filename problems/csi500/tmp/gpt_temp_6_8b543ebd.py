import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Fractal Efficiency Components
    # Price Path Efficiency
    price_cumulative_movement = data['close'].diff().abs().rolling(window=10, min_periods=1).sum()
    price_net_movement = (data['close'] - data['close'].shift(10)).abs()
    price_efficiency = price_net_movement / (price_cumulative_movement + 1e-8)
    
    # Volume Path Efficiency
    volume_cumulative_movement = data['volume'].diff().abs().rolling(window=10, min_periods=1).sum()
    volume_net_movement = (data['volume'] - data['volume'].shift(10)).abs()
    volume_efficiency = volume_net_movement / (volume_cumulative_movement + 1e-8)
    
    # 2. Compute Volume-Weighted Price Efficiency
    liquidity_efficiency = (data['close'] - data['open']) / (data['volume'] * (data['high'] - data['low']) / (data['close'] + 1e-8) + 1e-8)
    
    # 3. Calculate Multi-Scale Acceleration
    # Price Acceleration
    price_momentum_1d = data['close'].pct_change(1)
    price_momentum_5d = data['close'].pct_change(5)
    
    # Second-order price momentum
    price_accel_3d = price_momentum_1d.rolling(window=3).mean()
    price_accel_8d = price_momentum_1d.rolling(window=8).mean()
    price_accel_21d = price_momentum_1d.rolling(window=21).mean()
    price_acceleration = (price_accel_3d + price_accel_8d + price_accel_21d) / 3
    
    # Volume Acceleration
    volume_momentum_1d = data['volume'].pct_change(1)
    volume_momentum_5d = data['volume'].pct_change(5)
    
    # Second-order volume momentum
    volume_accel_1d_change = volume_momentum_1d.diff()
    volume_accel_5d_change = volume_momentum_5d.diff()
    volume_acceleration = (volume_accel_1d_change + volume_accel_5d_change) / 2
    
    # 4. Detect Efficiency-Volume Regime Patterns
    volume_regime = data['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Rolling correlation between efficiency and volume changes
    efficiency_volume_corr_5d = price_efficiency.rolling(window=5).corr(volume_momentum_1d)
    efficiency_volume_corr_15d = price_efficiency.rolling(window=15).corr(volume_momentum_1d)
    efficiency_volume_corr_30d = price_efficiency.rolling(window=30).corr(volume_momentum_1d)
    
    regime_efficiency_pattern = (efficiency_volume_corr_5d + efficiency_volume_corr_15d + efficiency_volume_corr_30d) / 3
    
    # 5. Measure Gap Absorption Context
    overnight_gap = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    intraday_range = (data['high'] - data['low']) / (data['close'] + 1e-8)
    gap_absorption_efficiency = intraday_range / (abs(overnight_gap) + 1e-8)
    
    # Gap direction persistence
    gap_direction = np.sign(overnight_gap)
    gap_persistence = gap_direction.rolling(window=3).sum()
    
    # 6. Detect Divergence Patterns
    # Efficiency Divergence
    efficiency_divergence = price_efficiency - volume_efficiency
    
    # Acceleration Divergence
    acceleration_divergence = price_acceleration - volume_acceleration
    
    # 7. Synthesize Composite Factor
    # Combine multi-scale signals
    divergence_product = efficiency_divergence * acceleration_divergence
    
    # Scale by liquidity efficiency momentum
    liquidity_momentum = liquidity_efficiency.rolling(window=5).mean()
    scaled_divergence = divergence_product * liquidity_momentum
    
    # Weight by gap absorption context
    gap_weight = gap_absorption_efficiency.rolling(window=5).mean()
    weighted_signal = scaled_divergence * gap_weight
    
    # Incorporate regime transitions
    regime_adjustment = regime_efficiency_pattern * volume_regime
    regime_adjusted_signal = weighted_signal * (1 + regime_adjustment)
    
    # Apply non-linear transformation
    final_signal = np.tanh(regime_adjusted_signal)
    
    # Maintain directional integrity from gap persistence
    directional_bias = np.tanh(gap_persistence / 3)
    final_factor = final_signal * (1 + 0.2 * directional_bias)
    
    return pd.Series(final_factor, index=data.index, name='multi_scale_efficiency_acceleration_divergence')
