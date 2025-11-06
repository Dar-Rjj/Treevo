import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-regime adaptive momentum convergence with volume-pressure confirmation.
    
    Interpretation:
    - Blends 3 ultra-short momentum signals: intraday, daily, and acceleration momentum
    - Uses volume-pressure ratio for immediate signal confirmation
    - Adapts momentum weights based on volatility regimes (high/low volatility environments)
    - Momentum convergence enhances signal when multiple timeframes align
    - Volume-pressure weighted momentum emphasizes high-activity periods
    - Volatility-normalized output provides robust, regime-agnostic signals
    - Positive values indicate strong bullish momentum with volume confirmation
    - Negative values suggest bearish pressure with distribution patterns
    """
    
    # Ultra-short momentum signals
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    daily_momentum = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    momentum_acceleration = intraday_momentum * daily_momentum
    
    # Volume-pressure confirmation
    volume_pressure = df['volume'] / (df['volume'].rolling(window=3).mean() + 1e-7)
    
    # Volatility regime detection (rolling standard deviation of returns)
    returns = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    volatility_regime = returns.rolling(window=5).std()
    high_vol_threshold = volatility_regime.quantile(0.7)
    low_vol_threshold = volatility_regime.quantile(0.3)
    
    # Adaptive weights based on volatility regimes
    def get_adaptive_weights(volatility):
        if volatility > high_vol_threshold:
            # High volatility: emphasize intraday momentum and reduce acceleration
            return [0.45, 0.35, 0.20]
        elif volatility < low_vol_threshold:
            # Low volatility: balanced approach with slight emphasis on acceleration
            return [0.30, 0.30, 0.40]
        else:
            # Normal volatility: balanced weights
            return [0.35, 0.35, 0.30]
    
    # Apply adaptive weights
    adaptive_alpha = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        vol = volatility_regime.loc[date] if pd.notna(volatility_regime.loc[date]) else np.nan
        if pd.notna(vol):
            weights = get_adaptive_weights(vol)
            intra = intraday_momentum.loc[date] if pd.notna(intraday_momentum.loc[date]) else 0
            daily = daily_momentum.loc[date] if pd.notna(daily_momentum.loc[date]) else 0
            accel = momentum_acceleration.loc[date] if pd.notna(momentum_acceleration.loc[date]) else 0
            
            adaptive_alpha.loc[date] = (
                weights[0] * intra +
                weights[1] * daily +
                weights[2] * accel
            )
    
    # Momentum convergence signal
    momentum_convergence = np.sign(intraday_momentum) * np.sign(daily_momentum) * (abs(intraday_momentum) + abs(daily_momentum))
    
    # Volume-pressure weighted final factor
    volume_weighted_factor = volume_pressure * adaptive_alpha
    
    # Volatility-normalized output for robustness
    volatility_normalizer = volatility_regime.rolling(window=10).mean() + 1e-7
    final_alpha = volume_weighted_factor * (1 + 0.15 * np.sign(momentum_convergence)) / volatility_normalizer
    
    return final_alpha
