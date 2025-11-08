import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adjusted Price-Volume Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Identify Volatility Regime
    # Calculate daily range
    daily_range = data['high'] - data['low']
    
    # Calculate 20-day rolling standard deviation of range
    range_volatility = daily_range.rolling(window=20, min_periods=10).std()
    
    # Classify regimes using historical percentiles (30-day lookback)
    volatility_regime = pd.Series(index=data.index, dtype='float64')
    for i in range(30, len(data)):
        if i < 30:
            volatility_regime.iloc[i] = 0
            continue
            
        current_vol = range_volatility.iloc[i]
        historical_vols = range_volatility.iloc[i-30:i]
        
        if len(historical_vols) > 0:
            vol_percentile = (historical_vols < current_vol).sum() / len(historical_vols)
            # 0 = low volatility, 1 = high volatility
            volatility_regime.iloc[i] = vol_percentile
        else:
            volatility_regime.iloc[i] = 0.5
    
    # 2. Calculate Price-Volume Divergence
    
    # Price Trend Component
    price_ema_10 = data['close'].ewm(span=10, adjust=False).mean()
    price_ema_20 = data['close'].ewm(span=20, adjust=False).mean()
    price_trend = (price_ema_10 - price_ema_20) / data['close']
    
    # Volume Trend Component
    volume_ema_10 = data['volume'].ewm(span=10, adjust=False).mean()
    volume_ema_20 = data['volume'].ewm(span=20, adjust=False).mean()
    volume_trend = (volume_ema_10 - volume_ema_20) / (data['volume'] + 1e-8)
    
    # Divergence Detection - rolling correlation
    divergence_strength = pd.Series(index=data.index, dtype='float64')
    for i in range(15, len(data)):
        if i < 15:
            divergence_strength.iloc[i] = 0
            continue
            
        # Calculate 15-day rolling correlation
        price_window = price_trend.iloc[i-14:i+1]
        volume_window = volume_trend.iloc[i-14:i+1]
        
        if len(price_window) >= 10 and price_window.std() > 0 and volume_window.std() > 0:
            correlation = price_window.corr(volume_window)
            # Negative correlation indicates divergence
            divergence_magnitude = -correlation if correlation < 0 else 0
            # Scale by the strength of individual trends
            trend_strength = np.sqrt(price_window.var() * volume_window.var())
            divergence_strength.iloc[i] = divergence_magnitude * trend_strength
        else:
            divergence_strength.iloc[i] = 0
    
    # 3. Regime-Adjusted Signal Combination
    
    factor = pd.Series(index=data.index, dtype='float64')
    
    for i in range(len(data)):
        if i < 30:  # Need enough data for regime calculation
            factor.iloc[i] = 0
            continue
            
        regime = volatility_regime.iloc[i]
        divergence = divergence_strength.iloc[i]
        current_vol = range_volatility.iloc[i]
        
        # Regime adjustment factor
        if regime < 0.3:  # Low volatility regime
            # Amplify signals in low volatility
            regime_factor = 1.5
            # Inverse volatility weighting
            vol_weight = 1.0 / (current_vol + 1e-8)
            regime_adjustment = regime_factor * min(vol_weight, 5.0)  # Cap the weighting
            
        elif regime > 0.7:  # High volatility regime
            # Suppress noise in high volatility
            regime_factor = 0.5
            # Volatility normalization
            vol_weight = 1.0 / (current_vol + 1e-8)
            regime_adjustment = regime_factor * min(vol_weight, 2.0)  # Conservative cap
            
        else:  # Normal volatility regime
            regime_adjustment = 1.0
        
        # Apply directional bias based on price trend
        price_direction = 1 if price_trend.iloc[i] > 0 else -1
        
        # Combine signals
        factor.iloc[i] = divergence * regime_adjustment * price_direction
    
    # Normalize the factor
    if len(factor) > 0:
        factor = (factor - factor.mean()) / (factor.std() + 1e-8)
    
    return factor
