import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Volatility Momentum
    # Multi-timeframe momentum alignment
    momentum_3d = data['close'].pct_change(3)
    momentum_10d = data['close'].pct_change(10)
    momentum_20d = data['close'].pct_change(20)
    
    # Volatility asymmetry calculation
    high_close_returns = (data['high'] - data['close'].shift(1)) / data['close'].shift(1)
    low_close_returns = (data['low'] - data['close'].shift(1)) / data['close'].shift(1)
    
    upside_vol = high_close_returns.rolling(window=10, min_periods=5).apply(
        lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 2 else 0
    )
    downside_vol = low_close_returns.rolling(window=10, min_periods=5).apply(
        lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 2 else 0
    )
    
    volatility_ratio = upside_vol / (downside_vol + 1e-8)
    volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Combined momentum alignment with volatility asymmetry
    aligned_momentum = (momentum_3d + momentum_10d + momentum_20d) / 3
    asym_vol_momentum = aligned_momentum * volatility_ratio
    
    # Volume-Clustered Range Breakout
    daily_range = data['high'] - data['low']
    avg_range_20d = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Resistance from past high volume days
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    high_volume_mask = data['volume'] > (1.5 * volume_20d_avg)
    
    # Calculate resistance levels from high volume days
    resistance_levels = data['high'].rolling(window=20, min_periods=5).apply(
        lambda x: np.mean(x[high_volume_mask.reindex(x.index).fillna(False)]) if high_volume_mask.reindex(x.index).sum() > 0 else np.nan
    )
    
    breakout_strength = (data['close'] - resistance_levels) / (avg_range_20d + 1e-8)
    normalized_range = daily_range / (avg_range_20d + 1e-8)
    
    # Volume cluster intensity
    volume_cluster_intensity = high_volume_mask.rolling(window=5, min_periods=3).sum() / 5
    volume_breakout_signal = breakout_strength * normalized_range * volume_cluster_intensity
    
    # Intraday Efficiency Persistence
    morning_strength = (data['high'] - data['open']) / data['open']
    afternoon_efficiency = (data['close'] - data['high']) / data['high']
    
    daily_efficiency_ratio = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    efficiency_5d_avg = daily_efficiency_ratio.rolling(window=5, min_periods=3).mean()
    
    # Momentum acceleration and consistency
    efficiency_acceleration = daily_efficiency_ratio - efficiency_5d_avg
    
    # Track consecutive same-direction patterns
    morning_direction = morning_strength > 0
    afternoon_direction = afternoon_efficiency > 0
    
    same_direction_mask = morning_direction == afternoon_direction
    consecutive_patterns = same_direction_mask.rolling(window=3, min_periods=2).sum()
    
    intraday_momentum = efficiency_acceleration * consecutive_patterns
    
    # Liquidity-Scaled Gap Reversion
    gap_pct = (data['open'] / data['close'].shift(1) - 1).fillna(0)
    
    # Effective spread proxy
    effective_spread = (data['amount'] / (data['volume'] + 1e-8)).fillna(0)
    spread_5d_avg = effective_spread.rolling(window=5, min_periods=3).mean()
    normalized_spread = effective_spread / (spread_5d_avg + 1e-8)
    
    # Reversion prediction based on gap magnitude
    gap_reversion = -gap_pct * np.abs(gap_pct)
    liquidity_reversion = gap_reversion * normalized_spread
    
    # Regime-Transition Order Flow
    daily_true_range = data['high'] - data['low']
    true_range_20d_avg = daily_true_range.rolling(window=20, min_periods=10).mean()
    
    # Volatility regime detection
    volatility_regime = daily_true_range > true_range_20d_avg
    regime_transition = (volatility_regime != volatility_regime.shift(1)) & volatility_regime
    
    # Cumulative signed volume
    daily_direction = data['close'] > data['open']
    signed_volume = data['volume'] * np.where(daily_direction, 1, -1)
    cumulative_signed_volume = signed_volume.rolling(window=10, min_periods=5).sum()
    
    # Regime-adaptive signals
    regime_change_magnitude = (daily_true_range - true_range_20d_avg) / (true_range_20d_avg + 1e-8)
    regime_order_flow = cumulative_signed_volume * regime_change_magnitude * regime_transition.astype(float)
    
    # Combine all factors with equal weighting
    factor_values = (
        asym_vol_momentum.fillna(0) +
        volume_breakout_signal.fillna(0) +
        intraday_momentum.fillna(0) +
        liquidity_reversion.fillna(0) +
        regime_order_flow.fillna(0)
    ) / 5
    
    return pd.Series(factor_values, index=data.index)
