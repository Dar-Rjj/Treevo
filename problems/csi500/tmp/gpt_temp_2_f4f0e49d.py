import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Momentum-Volume Divergence Alpha Factor
    Combines multi-timeframe momentum with volume divergence patterns and regime-aware weighting
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Price momentum
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Exponential Smoothing Application
    alpha = 0.3
    
    # Price momentum smoothing
    data['ema_price_5d'] = data['price_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    data['ema_price_10d'] = data['price_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    data['ema_price_20d'] = data['price_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Volume momentum smoothing
    data['ema_volume_5d'] = data['volume_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    data['ema_volume_10d'] = data['volume_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    data['ema_volume_20d'] = data['volume_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum Acceleration
    data['price_accel_5d'] = data['ema_price_5d'] - data['ema_price_5d'].shift(1)
    data['price_accel_10d'] = data['ema_price_10d'] - data['ema_price_10d'].shift(1)
    data['price_accel_20d'] = data['ema_price_20d'] - data['ema_price_20d'].shift(1)
    
    data['volume_accel_5d'] = data['ema_volume_5d'] - data['ema_volume_5d'].shift(1)
    data['volume_accel_10d'] = data['ema_volume_10d'] - data['ema_volume_10d'].shift(1)
    data['volume_accel_20d'] = data['ema_volume_20d'] - data['ema_volume_20d'].shift(1)
    
    # Regime-Aware Weighting System
    # Volatility regime detection
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_range_20d'] = data['daily_range'].rolling(window=20).mean()
    data['high_vol_regime'] = data['daily_range'] > data['avg_range_20d']
    
    # Adaptive weight assignment with 5-day persistence
    data['regime_weight'] = data['high_vol_regime'].rolling(window=5, min_periods=1).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else False)
    data['price_weight'] = np.where(data['regime_weight'], 0.3, 0.7)
    data['volume_weight'] = np.where(data['regime_weight'], 0.7, 0.3)
    
    # Divergence Pattern Recognition
    # Directional divergence signals
    data['divergence_5d'] = np.where(
        (data['ema_price_5d'] > 0) & (data['ema_volume_5d'] < 0), -1,  # Bearish divergence
        np.where((data['ema_price_5d'] < 0) & (data['ema_volume_5d'] > 0), 1, 0)  # Bullish divergence
    )
    
    data['divergence_10d'] = np.where(
        (data['ema_price_10d'] > 0) & (data['ema_volume_10d'] < 0), -1,
        np.where((data['ema_price_10d'] < 0) & (data['ema_volume_10d'] > 0), 1, 0)
    )
    
    data['divergence_20d'] = np.where(
        (data['ema_price_20d'] > 0) & (data['ema_volume_20d'] < 0), -1,
        np.where((data['ema_price_20d'] < 0) & (data['ema_volume_20d'] > 0), 1, 0)
    )
    
    # Magnitude divergence (strength of signals)
    data['magnitude_div_5d'] = np.abs(data['ema_price_5d']) - np.abs(data['ema_volume_5d'])
    data['magnitude_div_10d'] = np.abs(data['ema_price_10d']) - np.abs(data['ema_volume_10d'])
    data['magnitude_div_20d'] = np.abs(data['ema_price_20d']) - np.abs(data['ema_volume_20d'])
    
    # Factor Construction
    # Multi-timeframe weights
    timeframe_weights = {'5d': 0.4, '10d': 0.35, '20d': 0.25}
    
    # Signal strength calculation
    data['signal_5d'] = (
        data['divergence_5d'] * (1 + np.abs(data['magnitude_div_5d'])) +
        data['price_accel_5d'] * data['price_weight'] +
        data['volume_accel_5d'] * data['volume_weight']
    )
    
    data['signal_10d'] = (
        data['divergence_10d'] * (1 + np.abs(data['magnitude_div_10d'])) +
        data['price_accel_10d'] * data['price_weight'] +
        data['volume_accel_10d'] * data['volume_weight']
    )
    
    data['signal_20d'] = (
        data['divergence_20d'] * (1 + np.abs(data['magnitude_div_20d'])) +
        data['price_accel_20d'] * data['price_weight'] +
        data['volume_accel_20d'] * data['volume_weight']
    )
    
    # Combine signals with timeframe weights
    data['combined_signal'] = (
        data['signal_5d'] * timeframe_weights['5d'] +
        data['signal_10d'] * timeframe_weights['10d'] +
        data['signal_20d'] * timeframe_weights['20d']
    )
    
    # Volatility scaling
    data['alpha_factor'] = data['combined_signal'] * data['avg_range_20d']
    
    # Return the alpha factor series
    return data['alpha_factor']
