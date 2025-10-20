import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Momentum with Decay
    def calculate_decayed_momentum(close, n=20, decay_factor=0.9):
        returns = close.pct_change()
        weights = np.array([decay_factor ** i for i in range(n)])[::-1]
        weights = weights / weights.sum()
        
        decayed_momentum = pd.Series(index=close.index, dtype=float)
        for i in range(n-1, len(close)):
            if i >= n-1:
                window_returns = returns.iloc[i-n+1:i+1]
                decayed_momentum.iloc[i] = (window_returns * weights).sum()
        
        return decayed_momentum
    
    # Assess Momentum Quality
    def assess_momentum_quality(close, volume, n=10):
        returns = close.pct_change()
        
        # Momentum sustainability (consistency of returns)
        momentum_consistency = returns.rolling(window=n).apply(
            lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
        )
        
        # Volume support (correlation between volume and absolute returns)
        volume_support = volume.rolling(window=n).corr(returns.abs())
        
        # Combined quality score
        quality_score = 0.6 * momentum_consistency + 0.4 * volume_support.fillna(0)
        return quality_score
    
    # Measure Volatility Regime
    def measure_volatility_regime(close, vol_window=30, regime_window=10):
        returns = close.pct_change()
        rolling_vol = returns.rolling(window=vol_window).std()
        
        # Identify volatility regimes (high/low relative to recent history)
        vol_regime = (rolling_vol / rolling_vol.rolling(window=regime_window).mean() - 1)
        
        # Smooth the regime indicator
        vol_regime_smooth = vol_regime.rolling(window=5).mean()
        return vol_regime_smooth
    
    # Incorporate volume acceleration
    def calculate_volume_acceleration(volume, n=5):
        volume_ma = volume.rolling(window=n).mean()
        volume_accel = volume_ma.pct_change(periods=3)
        return volume_accel
    
    # Main factor calculation
    close = df['close']
    volume = df['volume']
    
    # Calculate components
    decayed_momentum = calculate_decayed_momentum(close, n=20, decay_factor=0.9)
    momentum_quality = assess_momentum_quality(close, volume, n=10)
    vol_regime = measure_volatility_regime(close, vol_window=30, regime_window=10)
    volume_accel = calculate_volume_acceleration(volume, n=5)
    
    # Combine components
    # Base composite: decayed momentum adjusted by quality
    base_composite = decayed_momentum * momentum_quality
    
    # Volatility regime adjustment (inverse relationship - reduce position in high vol)
    vol_adjustment = 1 / (1 + np.exp(vol_regime * 2))
    
    # Final factor with volume acceleration boost
    factor = base_composite * vol_adjustment * (1 + volume_accel.fillna(0))
    
    return factor
