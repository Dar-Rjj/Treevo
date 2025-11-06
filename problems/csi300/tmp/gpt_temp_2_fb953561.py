import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence
    """
    # Volatility-Normalized Momentum
    # Short-term momentum (5-day close returns)
    momentum_5d = df['close'].pct_change(5)
    
    # Daily volatility (high - low)
    daily_vol = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Volatility-normalized momentum
    vol_norm_momentum = momentum_5d / daily_vol.rolling(5).mean()
    
    # Volume Confirmation
    # Volume trend (5-day volume slope)
    volume_trend = df['volume'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Price-volume alignment check
    price_trend = df['close'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Volume divergence flag (opposing directions)
    volume_divergence = np.sign(price_trend) != np.sign(volume_trend)
    
    # Volume alignment multiplier
    volume_multiplier = np.where(volume_divergence, -0.5, 1.0)
    
    # Regime Detection
    # Volatility regime (20-day ATR)
    atr_20d = (
        (df['high'] - df['low']).rolling(20).mean() / 
        df['close'].rolling(20).mean()
    )
    
    # High volatility: mean reversion bias
    # Low volatility: momentum continuation bias
    volatility_regime = atr_20d.rolling(20).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else 
                  (-1 if x.iloc[-1] < x.quantile(0.3) else 0)
    )
    
    # Regime weights
    regime_weight = np.where(volatility_regime == 1, -0.8,  # High vol: mean reversion
                            np.where(volatility_regime == -1, 1.2,  # Low vol: momentum
                                    1.0))  # Normal regime
    
    # Alpha Combination
    # Base signal: volatility-normalized momentum
    base_signal = vol_norm_momentum
    
    # Volume-confirmed signal: base + volume alignment
    volume_confirmed_signal = base_signal * volume_multiplier
    
    # Final alpha: regime-weighted combination
    final_alpha = volume_confirmed_signal * regime_weight
    
    return final_alpha
