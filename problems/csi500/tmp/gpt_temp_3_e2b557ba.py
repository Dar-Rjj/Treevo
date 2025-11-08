import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum
    close = df['close']
    
    # Calculate momentum rates of change
    mom_short = close / close.shift(5) - 1
    mom_medium = close / close.shift(20) - 1
    mom_long = close / close.shift(60) - 1
    
    # Volatility Normalization
    # High-Low Range Volatility
    hl_range = df['high'] - df['low']
    hl_volatility = hl_range.rolling(window=20).mean()
    
    # Returns Volatility
    daily_returns = close.pct_change()
    returns_volatility = daily_returns.rolling(window=20).std()
    
    # Combined volatility measure
    combined_volatility = (hl_volatility + returns_volatility) / 2
    
    # Normalize momentum by volatility
    mom_short_norm = mom_short / combined_volatility
    mom_medium_norm = mom_medium / combined_volatility
    mom_long_norm = mom_long / combined_volatility
    
    # Volume Confirmation
    volume = df['volume']
    
    # Volume moving averages
    vol_ma_short = volume.rolling(window=5).mean()
    vol_ma_long = volume.rolling(window=20).mean()
    
    # Volume ratio and intensity
    volume_ratio = vol_ma_short / vol_ma_long
    volume_intensity = volume / volume.shift(1)
    
    # Volume strength multiplier
    volume_strength = volume_ratio * volume_intensity
    
    # Combine normalized momentum with volume confirmation
    combined_signal = (mom_short_norm * 0.4 + 
                      mom_medium_norm * 0.35 + 
                      mom_long_norm * 0.25) * volume_strength
    
    # Regime Detection
    # Volatility regime - ATR based
    atr = (df['high'] - df['low']).rolling(window=20).mean()
    atr_median = atr.rolling(window=60).median()
    volatility_regime = atr / atr_median
    
    # Trend regime
    ma_20 = close.rolling(window=20).mean()
    ma_50 = close.rolling(window=50).mean()
    
    # Price position relative to moving averages
    price_above_ma = (close > ma_20).astype(int) + (close > ma_50).astype(int)
    
    # Trend slope
    trend_slope = close.rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )
    
    # Trend regime score
    trend_regime = price_above_ma + np.sign(trend_slope)
    
    # Regime-based scaling
    # High volatility: reduce factor magnitude
    volatility_scale = np.where(volatility_regime > 1.2, 0.7, 
                               np.where(volatility_regime < 0.8, 1.2, 1.0))
    
    # Trend-based scaling
    trend_scale = np.where(trend_regime >= 2, 1.3,  # Strong uptrend
                          np.where(trend_regime <= 0, 0.7,  # Downtrend
                                  1.0))  # Neutral
    
    # Apply regime scaling
    regime_scaled_signal = combined_signal * volatility_scale * trend_scale
    
    # Recursive Smoothing with Exponential Weighting
    factor = pd.Series(index=df.index, dtype=float)
    decay = 0.9
    
    for i in range(len(regime_scaled_signal)):
        if i == 0:
            factor.iloc[i] = regime_scaled_signal.iloc[i]
        else:
            factor.iloc[i] = (decay * factor.iloc[i-1] + 
                             (1 - decay) * regime_scaled_signal.iloc[i])
    
    return factor
