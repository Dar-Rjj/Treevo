import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Efficiency Momentum with Volume-Price Regime Alignment
    """
    data = df.copy()
    
    # Intraday Range Efficiency Analysis
    # Daily Range Utilization
    daily_range = data['high'] - data['low']
    daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
    range_efficiency = (data['close'] - data['open']) / daily_range
    
    # Multi-day Efficiency Persistence (5-day)
    efficiency_5d = range_efficiency.rolling(window=5, min_periods=3).mean()
    efficiency_direction = np.sign(range_efficiency.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(np.diff(np.sign(x.dropna())) == 0) if len(x.dropna()) > 1 else 0
    ))
    
    # Efficiency Regime Detection (10-day pattern)
    efficiency_10d_avg = range_efficiency.rolling(window=10, min_periods=5).mean()
    efficiency_regime = range_efficiency - efficiency_10d_avg
    
    # Volume-Price Momentum Divergence
    # Price Momentum Patterns
    price_momentum_3d = data['close'] / data['close'].shift(3) - 1
    price_momentum_8d = data['close'] / data['close'].shift(8) - 1
    
    # Volume Momentum Context
    volume_momentum_3d = data['volume'] / data['volume'].shift(3)
    volume_momentum_10d = data['volume'] / data['volume'].shift(10)
    volume_pattern_ratio = volume_momentum_3d / volume_momentum_10d.replace(0, np.nan)
    
    # Price-Volume Divergence
    price_volume_divergence = np.sign(price_momentum_3d) * np.sign(volume_momentum_3d - 1)
    divergence_magnitude = abs(price_momentum_3d) * abs(volume_momentum_3d - 1)
    divergence_consistency = price_volume_divergence.rolling(window=5, min_periods=3).mean()
    
    # Volatility-Regime Breakout Quality
    # Range Breakout Candidates (15-day)
    rolling_high_15d = data['high'].rolling(window=15, min_periods=10).max()
    rolling_low_15d = data['low'].rolling(window=15, min_periods=10).min()
    
    upper_breakout = (data['high'] > rolling_high_15d.shift(1)).astype(int)
    lower_breakout = (data['low'] < rolling_low_15d.shift(1)).astype(int)
    breakout_direction = upper_breakout - lower_breakout
    breakout_magnitude = np.where(
        upper_breakout == 1, 
        (data['high'] - rolling_high_15d.shift(1)) / rolling_high_15d.shift(1),
        np.where(
            lower_breakout == 1,
            (rolling_low_15d.shift(1) - data['low']) / rolling_low_15d.shift(1),
            0
        )
    )
    
    # Breakout Volume Support
    avg_volume_10d = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_support = data['volume'] / avg_volume_10d.replace(0, np.nan)
    volume_range_ratio = data['volume'] / daily_range.replace(0, np.nan)
    
    # Volatility Context Analysis
    returns = data['close'].pct_change()
    volatility_15d = returns.rolling(window=15, min_periods=10).std()
    volatility_regime = volatility_15d > volatility_15d.rolling(window=30, min_periods=15).median()
    
    # Opening Gap Momentum Analysis
    gap_magnitude = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap distribution percentile (20-day)
    gap_percentile = gap_magnitude.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x.dropna()) > 0 else 0.5
    )
    
    # Volume Momentum Confirmation
    volume_acceleration_5d = data['volume'] / data['volume'].shift(5)
    gap_volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Composite Alpha Factor Generation
    # Component Integration and Weighting
    
    # Range Efficiency Component
    efficiency_component = (efficiency_5d * efficiency_direction * efficiency_regime)
    
    # Price-Volume Divergence Component
    pv_divergence_component = (price_volume_divergence * divergence_magnitude * divergence_consistency)
    
    # Breakout Quality Component
    breakout_component = (breakout_direction * breakout_magnitude * volume_support * volume_range_ratio)
    
    # Gap Momentum Component
    gap_component = (gap_magnitude * gap_percentile * volume_acceleration_5d * gap_volume_ratio)
    
    # Regime-Adaptive Signal Synthesis
    # Volatility regime-dependent weighting
    low_vol_weight = 0.7
    high_vol_weight = 1.3
    
    volatility_weight = np.where(volatility_regime, high_vol_weight, low_vol_weight)
    
    # Volume momentum context adjustment
    volume_context = np.where(volume_momentum_3d > 1, 1.2, 0.8)
    
    # Efficiency persistence scaling
    efficiency_persistence = efficiency_direction.rolling(window=3, min_periods=2).mean()
    
    # Final composite factor
    composite_factor = (
        efficiency_component * pv_divergence_component * 
        breakout_component * gap_component * 
        volatility_weight * volume_context * efficiency_persistence
    )
    
    # Normalize and clean
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    composite_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=10).mean()) / composite_factor.rolling(window=20, min_periods=10).std()
    
    return pd.Series(composite_factor, index=data.index)
