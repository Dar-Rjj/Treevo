import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Divergence with Volume Efficiency
    """
    # Calculate Daily Returns
    returns = df['close'].pct_change()
    
    # Compute Rolling Volatility (20-day)
    volatility = returns.rolling(window=20, min_periods=10).std()
    
    # Classify Volatility Regimes using 60-day historical percentiles
    vol_percentile = volatility.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) * 2 + 
                 (x.iloc[-1] < np.percentile(x.dropna(), 20)) * 1, 
        raw=False
    )
    
    # Regime classification
    high_vol_regime = (vol_percentile == 2)
    low_vol_regime = (vol_percentile == 1)
    normal_vol_regime = (~high_vol_regime & ~low_vol_regime)
    
    # Compute Multi-Timeframe Momentum
    mom_short = df['close'] / df['close'].shift(5) - 1
    mom_medium = df['close'] / df['close'].shift(15) - 1
    mom_long = df['close'] / df['close'].shift(20) - 1
    
    # Calculate Regime-Adaptive Divergence
    regime_divergence = pd.Series(index=df.index, dtype=float)
    
    # High Volatility: Focus on short-term acceleration
    high_vol_div = (
        0.7 * (mom_short - mom_medium) + 
        0.3 * (mom_short - mom_long)
    )
    
    # Low Volatility: Focus on sustained trends
    low_vol_div = (
        0.8 * (mom_medium - mom_long) + 
        0.2 * mom_short
    )
    
    # Normal Volatility: Balanced approach
    normal_vol_div = (
        (mom_short - mom_medium) + 
        (mom_short - mom_long) + 
        (mom_medium - mom_long)
    ) / 3
    
    # Apply regime-specific divergence
    regime_divergence[high_vol_regime] = high_vol_div[high_vol_regime]
    regime_divergence[low_vol_regime] = low_vol_div[low_vol_regime]
    regime_divergence[normal_vol_regime] = normal_vol_div[normal_vol_regime]
    
    # Calculate Volume-Pressure Asymmetry
    up_days = df['close'] > df['close'].shift(1)
    down_days = df['close'] < df['close'].shift(1)
    
    up_volume_pressure = df['volume'].where(up_days, 0).rolling(window=10, min_periods=5).sum()
    down_volume_pressure = df['volume'].where(down_days, 0).rolling(window=10, min_periods=5).sum()
    
    # Avoid division by zero
    pressure_ratio = up_volume_pressure / (down_volume_pressure + 1e-8)
    
    # Calculate Volume Acceleration Efficiency
    volume_momentum = df['volume'] / df['volume'].rolling(window=5, min_periods=3).mean()
    volume_acceleration = volume_momentum / volume_momentum.shift(3) - 1
    
    volume_efficiency = volume_acceleration * pressure_ratio
    
    # Combine with Range Efficiency
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Intraday Efficiency Ratio
    daily_return_abs = abs(df['close'] / df['close'].shift(1) - 1)
    efficiency_ratio = daily_return_abs / (true_range + 1e-8)
    
    # Final Volume Efficiency Pressure
    volume_efficiency_pressure = volume_efficiency * efficiency_ratio
    
    # Detect Regime Transition Signals
    # Volatility Breakout
    vol_5day = returns.rolling(window=5, min_periods=3).std()
    vol_15day = returns.rolling(window=15, min_periods=10).std()
    vol_ratio = vol_5day / (vol_15day + 1e-8)
    
    volatility_breakout = pd.Series(0, index=df.index)
    volatility_breakout[vol_ratio > 2.0] = 1
    volatility_breakout[vol_ratio < 0.5] = -1
    
    # Momentum-Velocity Divergence
    # Calculate ATR for normalization
    atr_3day = true_range.rolling(window=3, min_periods=2).mean()
    price_velocity = (df['close'] - df['close'].shift(3)) / (atr_3day + 1e-8)
    
    momentum_velocity_div = price_velocity - regime_divergence
    
    # Generate Transition Score
    transition_score = (
        0.6 * volatility_breakout + 
        0.4 * np.sign(momentum_velocity_div)
    )
    
    # Synthesize Final Alpha Factor
    raw_factor = regime_divergence * volume_efficiency_pressure * (1 + 0.5 * transition_score)
    
    # Apply Dynamic Smoothing based on regime
    final_factor = pd.Series(index=df.index, dtype=float)
    
    # High Volatility: 3-day window
    final_factor[high_vol_regime] = raw_factor[high_vol_regime].rolling(
        window=3, min_periods=2
    ).mean()
    
    # Low Volatility: 8-day window  
    final_factor[low_vol_regime] = raw_factor[low_vol_regime].rolling(
        window=8, min_periods=5
    ).mean()
    
    # Normal Volatility: 5-day window
    final_factor[normal_vol_regime] = raw_factor[normal_vol_regime].rolling(
        window=5, min_periods=3
    ).mean()
    
    return final_factor
