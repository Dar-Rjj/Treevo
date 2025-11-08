import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum Divergence with Volume Confirmation
    """
    df = data.copy()
    
    # Compute daily returns for volatility calculations
    df['daily_return'] = df['close'].pct_change()
    
    # Asymmetric Timeframe Momentum Signals
    # Short-term Momentum (3-day)
    short_return = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    short_vol = df['daily_return'].rolling(window=5, min_periods=3).std()
    short_momentum = short_return / short_vol.replace(0, np.nan)
    
    # Medium-term Momentum (8-day)
    medium_return = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    medium_vol = df['daily_return'].rolling(window=10, min_periods=5).std()
    medium_momentum = medium_return / medium_vol.replace(0, np.nan)
    
    # Long-term Momentum (20-day)
    long_return = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    long_vol = df['daily_return'].rolling(window=15, min_periods=8).std()
    long_momentum = long_return / long_vol.replace(0, np.nan)
    
    # Volume Regime Confirmation
    # Compute Volume Percentile Bands
    vol_20_pct = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    vol_50_pct = df['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Calculate divergence and apply nonlinear transform
    vol_divergence = vol_20_pct - vol_50_pct
    vol_confirmation = np.sign(vol_divergence) * (np.abs(vol_divergence) ** (1/3))
    vol_confirmation = np.clip(vol_confirmation, -1, 1)
    
    # Momentum Divergence Factor
    # Calculate momentum spreads
    short_medium_spread = short_momentum - medium_momentum
    medium_long_spread = medium_momentum - long_momentum
    
    # Combine divergences multiplicatively and apply cube root
    momentum_divergence = (short_medium_spread * medium_long_spread) ** (1/3)
    
    # Final Alpha Combination
    # Multiply Momentum Divergence by Volume Confirmation
    raw_alpha = momentum_divergence * vol_confirmation
    
    # Regime-aware weighting using volatility
    vol_20_days = df['daily_return'].rolling(window=20, min_periods=10).std()
    vol_percentile = vol_20_days.rolling(window=50, min_periods=25).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Apply regime-aware scaling
    # High volatility: scale down, Low volatility: scale up
    regime_weight = 1.0 - vol_percentile  # Inverse weighting
    final_alpha = raw_alpha * regime_weight
    
    # Bound final output
    final_alpha = np.clip(final_alpha, -2, 2)
    
    return final_alpha
