import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence via percentile-based regime weights.
    
    Interpretation:
    - Momentum acceleration hierarchy across intraday, overnight, and daily timeframes
    - Volume divergence detection using percentile-based regime classification
    - Smooth regime transitions via exponential weighting and multiplicative combinations
    - Dynamic regime persistence enhances signal stability during consistent market conditions
    - Volume-momentum synchronization improves signal reliability across different market states
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest deteriorating momentum with volume distribution patterns
    """
    
    # Core momentum components with acceleration hierarchy
    intraday_return = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daily_return = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    
    # Momentum acceleration using exponential smoothing
    intraday_accel = intraday_return - intraday_return.ewm(span=3).mean()
    overnight_accel = overnight_return - overnight_return.ewm(span=3).mean()
    daily_accel = daily_return - daily_return.ewm(span=5).mean()
    
    # Volume divergence detection using percentile-based regimes
    volume_ratio = df['volume'] / (df['volume'].rolling(window=10, min_periods=5).mean() + 1e-7)
    volume_percentile = volume_ratio.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1
    )
    
    # Volume regime classification with smooth transitions
    volume_regime = np.where(volume_percentile == 3, 2.0,  # High volume regime
                            np.where(volume_percentile == 2, 1.2,  # Medium volume regime
                                    np.where(volume_percentile == 1, 0.8,  # Low volume regime
                                            1.0)))  # Normal volume regime
    
    # Momentum regime persistence using rolling correlation
    momentum_persistence = (
        intraday_return.rolling(window=5).corr(daily_return) + 
        overnight_return.rolling(window=5).corr(daily_return)
    ) / 2
    
    # Multi-timeframe momentum convergence with regime weights
    momentum_convergence = (
        intraday_accel * volume_regime * 
        np.sign(intraday_accel * overnight_accel) *
        (1 + momentum_persistence.fillna(0))
    )
    
    # Volume-momentum synchronization factor
    volume_momentum_sync = (
        volume_regime * daily_accel * 
        np.sign(daily_accel * intraday_accel) *
        np.exp(-abs(momentum_persistence.fillna(0)))
    )
    
    # Combined alpha factor with multiplicative regime adaptation
    alpha_factor = (
        momentum_convergence * 
        volume_momentum_sync * 
        (1 + 0.1 * momentum_persistence.fillna(0))
    )
    
    return alpha_factor
