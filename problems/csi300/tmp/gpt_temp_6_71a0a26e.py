import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor combining regime-aware momentum acceleration with volatility-scaled 
    volume confirmation and nonlinear price efficiency. Captures stocks with accelerating 
    momentum in low-volatility regimes supported by volume trends and price efficiency.
    """
    
    # Regime-aware momentum acceleration with multiple timeframes
    momentum_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Nonlinear momentum acceleration combining short and medium-term trends
    short_term_accel = np.tanh(momentum_2d - momentum_5d)
    medium_term_accel = np.arctan(momentum_5d - momentum_10d)
    regime_aware_momentum = short_term_accel * medium_term_accel
    
    # Volatility-scaled components with regime detection
    returns = df['close'].pct_change()
    short_vol = returns.rolling(window=3).std()
    medium_vol = returns.rolling(window=8).std()
    vol_regime = np.arctan(short_vol / (medium_vol + 1e-7))
    
    # Volume confirmation with nonlinear transforms and trend alignment
    volume_momentum = df['volume'].pct_change(periods=3)
    volume_acceleration = volume_momentum - volume_momentum.shift(2)
    volume_trend_strength = np.sign(volume_acceleration) * np.sqrt(abs(volume_acceleration))
    
    # Price efficiency with range-based nonlinear scaling
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    directional_efficiency = (df['close'] - df['open']) / (true_range + 1e-7)
    efficiency_persistence = directional_efficiency.rolling(window=4).apply(
        lambda x: np.mean(np.sign(x) * np.sqrt(abs(x))))
    
    # Amount-based liquidity confirmation
    amount_trend = df['amount'].pct_change(periods=2)
    liquidity_momentum = np.arctan(amount_trend) * np.sign(amount_trend)
    
    # Regime-aware combination with nonlinear alignment
    momentum_component = regime_aware_momentum * (1 - abs(vol_regime))
    volume_component = volume_trend_strength * efficiency_persistence
    liquidity_component = np.tanh(liquidity_momentum) * np.sign(regime_aware_momentum)
    
    # Final alpha factor with cross-component validation
    alpha_factor = (
        momentum_component * 
        np.sign(volume_component) * np.log1p(abs(volume_component)) * 
        liquidity_component
    )
    
    return alpha_factor
