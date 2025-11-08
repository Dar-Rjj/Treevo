import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum Divergence Alpha Factor
    Combines multi-timeframe momentum with volume-price divergence patterns
    and regime-adaptive weighting for enhanced predictive power.
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    periods = [3, 8, 15]
    
    # Calculate price and volume momentum for each timeframe
    price_momentum = {}
    volume_momentum = {}
    
    for period in periods:
        price_momentum[period] = (df['close'] / df['close'].shift(period)) - 1
        volume_momentum[period] = (df['volume'] / df['volume'].shift(period)) - 1
    
    # Exponential Smoothing Application
    alpha = 0.4
    
    # Apply EMA to momentum series
    ema_price = {}
    ema_volume = {}
    
    for period in periods:
        ema_price[period] = price_momentum[period].ewm(alpha=alpha, adjust=False).mean()
        ema_volume[period] = volume_momentum[period].ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate Momentum Acceleration
    price_acceleration = {}
    volume_acceleration = {}
    
    for period in periods:
        price_acceleration[period] = ema_price[period] - ema_price[period].shift(1)
        volume_acceleration[period] = ema_volume[period] - ema_volume[period].shift(1)
    
    # Divergence Pattern Recognition
    directional_divergence = {}
    acceleration_divergence = {}
    
    for period in periods:
        # Directional divergence: sign mismatch between price and volume momentum
        price_dir = np.sign(ema_price[period])
        volume_dir = np.sign(ema_volume[period])
        directional_divergence[period] = (price_dir != volume_dir).astype(int) * np.abs(ema_price[period])
        
        # Acceleration divergence: difference between price and volume acceleration
        acceleration_divergence[period] = price_acceleration[period] - volume_acceleration[period]
    
    # Multi-timeframe consistency
    directional_consistency = pd.DataFrame(directional_divergence).mean(axis=1)
    acceleration_consistency = pd.DataFrame(acceleration_divergence).mean(axis=1)
    
    # Regime-Aware Weighting System
    # Calculate Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_20 = true_range.rolling(window=20).mean()
    
    # Volatility regime detection
    volatility_regime = (true_range > atr_20).astype(int)  # 1 for high, 0 for low
    
    # Adaptive Signal Weighting
    directional_signal = directional_consistency.copy()
    acceleration_signal = acceleration_consistency.copy()
    
    # Apply regime-adaptive weights
    for i in range(len(df)):
        if volatility_regime.iloc[i] == 1:  # High volatility regime
            weight_directional = 0.6
            weight_acceleration = 0.4
        else:  # Low volatility regime
            weight_directional = 0.4
            weight_acceleration = 0.6
        
        directional_signal.iloc[i] = directional_signal.iloc[i] * weight_directional
        acceleration_signal.iloc[i] = acceleration_signal.iloc[i] * weight_acceleration
    
    # Factor Synthesis
    composite_score = directional_signal + acceleration_signal
    
    # Signal Persistence Check
    signal_persistence = composite_score.rolling(window=2).apply(
        lambda x: 1 if len(x) == 2 and np.sign(x[0]) == np.sign(x[1]) else 0, 
        raw=True
    )
    
    # Apply persistence filter and calculate final alpha
    final_alpha = composite_score * signal_persistence
    
    # Normalize the final alpha
    final_alpha = (final_alpha - final_alpha.mean()) / final_alpha.std()
    
    return final_alpha
