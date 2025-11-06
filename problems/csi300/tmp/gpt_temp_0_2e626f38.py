import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum Acceleration with Volume Divergence factor
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Acceleration
    # Short-Term Momentum (1-5 days)
    mom_1d = data['close'].pct_change(1)
    mom_3d = data['close'].pct_change(3)
    mom_5d = data['close'].pct_change(5)
    
    # Medium-Term Momentum (5-10 days)
    # Calculate trend slopes using linear regression coefficients
    def rolling_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    mom_5d_slope = rolling_slope(data['close'], 5)
    mom_10d_slope = rolling_slope(data['close'], 10)
    
    # Momentum Acceleration - Rate of momentum change
    mom_accel_short = (mom_3d - mom_1d.shift(2)) / 2  # 3-day vs 1-day momentum
    mom_accel_medium = (mom_10d_slope - mom_5d_slope.shift(5)) / 5  # 10-day vs 5-day slope
    
    # Combined momentum acceleration
    momentum_acceleration = (mom_accel_short.rank() + mom_accel_medium.rank()) / 2
    
    # Volatility Regime Assessment
    # True Range Volatility
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    volatility_5d = true_range.rolling(window=5, min_periods=3).mean()
    volatility_20d = true_range.rolling(window=20, min_periods=15).mean()
    
    # Market Regime Classification
    volatility_ratio = volatility_5d / volatility_20d
    high_vol_regime = (volatility_ratio > 1.2).astype(int)
    low_vol_regime = (volatility_ratio < 0.8).astype(int)
    
    # Volume-Price Divergence
    # Short-Term Volume Trend (5 days)
    volume_5d_ma = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_5d_roc = volume_5d_ma.pct_change(3)
    
    # Long-Term Volume Trend (20 days)
    volume_20d_ma = data['volume'].rolling(window=20, min_periods=15).mean()
    volume_20d_roc = volume_20d_ma.pct_change(10)
    
    # Volume Divergence Factor
    volume_divergence = volume_5d_roc / (volume_20d_roc + 1e-8)
    
    # Adaptive Signal Generation
    # Regime-Dependent Parameters
    # Volatility-based lookback adjustment
    def adaptive_lookback(vol_regime):
        return np.where(vol_regime == 1, 3, 5)  # Shorter lookback in high vol
    
    # Momentum × Volume Interaction
    momentum_volume_interaction = momentum_acceleration * volume_divergence
    
    # Volatility-Regime Weighting
    # Dynamic signal scaling based on volatility regime
    regime_weight = np.where(high_vol_regime == 1, 0.7, 
                           np.where(low_vol_regime == 1, 1.3, 1.0))
    
    # Final factor calculation
    factor = momentum_volume_interaction * regime_weight
    
    # Clean and normalize
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / factor.rolling(window=20, min_periods=10).std()
    
    return factor
