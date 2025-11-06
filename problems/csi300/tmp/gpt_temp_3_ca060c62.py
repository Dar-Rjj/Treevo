import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-scaled momentum acceleration with volume divergence and regime-aware weighting.
    
    Interpretation:
    - Momentum acceleration captures rate of change in price movements across multiple timeframes
    - Volatility scaling adapts signal strength to current market conditions
    - Volume divergence identifies when trading activity confirms or contradicts price momentum
    - Regime-aware weighting emphasizes different momentum components based on volatility environment
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest accelerating bearish pressure with volume distribution
    """
    
    # Core momentum components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum acceleration (rate of change)
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(1)
    
    # Volatility estimation using daily range
    daily_range = (df['high'] - df['low']) / df['close']
    vol_5d = daily_range.rolling(window=5).std()
    
    # Volume divergence: current volume vs recent average
    volume_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_divergence = volume_ratio - volume_ratio.shift(1)
    
    # Regime detection based on volatility levels
    vol_regime = vol_5d.rolling(window=10).apply(lambda x: np.percentile(x, 70))
    high_vol_regime = vol_5d > vol_regime
    low_vol_regime = vol_5d < vol_5d.rolling(window=10).apply(lambda x: np.percentile(x, 30))
    
    # Volatility scaling factor
    vol_scale = 1.0 / (vol_5d + 1e-7)
    
    # Regime-aware weighting
    intraday_weight = np.where(high_vol_regime, 0.4, np.where(low_vol_regime, 0.2, 0.3))
    overnight_weight = np.where(high_vol_regime, 0.2, np.where(low_vol_regime, 0.3, 0.25))
    daily_weight = np.where(high_vol_regime, 0.2, np.where(low_vol_regime, 0.3, 0.25))
    volume_weight = np.where(high_vol_regime, 0.2, np.where(low_vol_regime, 0.2, 0.2))
    
    # Combine components with regime-aware weighting and volatility scaling
    momentum_component = (
        intraday_weight * intraday_accel +
        overnight_weight * overnight_accel +
        daily_weight * daily_accel
    ) * vol_scale
    
    volume_component = volume_weight * volume_divergence * np.sign(momentum_component)
    
    # Final alpha factor
    alpha_factor = momentum_component + volume_component
    
    return alpha_factor
