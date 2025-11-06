import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Weighted Momentum Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns and True Range
    data['returns'] = data['close'].pct_change()
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Acceleration
    data['momentum_5d_accel'] = data['momentum_5d'] / data['momentum_5d'].shift(1) - 1
    data['momentum_10d_accel'] = data['momentum_10d'] / data['momentum_10d'].shift(1) - 1
    data['momentum_20d_accel'] = data['momentum_20d'] / data['momentum_20d'].shift(1) - 1
    
    # Asymmetric Volatility Calculation
    data['daily_range'] = data['high'] - data['low']
    data['volatility_20d'] = data['daily_range'].rolling(window=20).mean()
    
    # Separate positive and negative return days for asymmetric volatility
    data['positive_day'] = (data['returns'] > 0).astype(int)
    data['negative_day'] = (data['returns'] < 0).astype(int)
    
    # Volatility regime detection
    data['vol_regime'] = np.where(
        data['daily_range'] > data['volatility_20d'] * 1.2, 2,  # High volatility
        np.where(data['daily_range'] < data['volatility_20d'] * 0.8, 0, 1)  # Low volatility
    )
    
    # Volatility-Weighted Momentum Adjustment
    # Higher weight for low volatility down days, lower weight for high volatility up days
    volatility_weights = np.where(
        (data['vol_regime'] == 0) & (data['negative_day'] == 1), 1.5,  # Low vol down days
        np.where(
            (data['vol_regime'] == 2) & (data['positive_day'] == 1), 0.5,  # High vol up days
            1.0  # Standard weighting
        )
    )
    
    data['vol_weighted_momentum_5d'] = data['momentum_5d'] * volatility_weights
    data['vol_weighted_momentum_10d'] = data['momentum_10d'] * volatility_weights
    data['vol_weighted_momentum_20d'] = data['momentum_20d'] * volatility_weights
    
    # Volatility-weighted acceleration
    data['vol_weighted_accel_5d'] = data['momentum_5d_accel'] * volatility_weights
    data['vol_weighted_accel_20d'] = data['momentum_20d_accel'] * volatility_weights
    
    # Momentum Divergence Analysis
    data['acceleration_divergence'] = data['vol_weighted_accel_5d'] - data['vol_weighted_accel_20d']
    data['momentum_spread'] = data['vol_weighted_momentum_5d'] - data['vol_weighted_momentum_20d']
    
    # Divergence persistence
    data['divergence_persistence'] = data['acceleration_divergence'].rolling(
        window=5, min_periods=3
    ).apply(lambda x: np.sum(x > 0) if len(x) >= 3 else np.nan)
    
    # Volume-Confirmed Divergence Signals
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_trend'] = np.where(
        data['volume_ma_5d'] / data['volume_ma_20d'] > 1, 1, -1
    )
    
    # Volume acceleration relative to volatility
    data['volume_accel'] = data['volume_ma_5d'] / data['volume_ma_5d'].shift(1) - 1
    data['vol_adj_volume_accel'] = data['volume_accel'] / (data['volatility_20d'] + 1e-8)
    
    # Volume-volatility integration
    volume_confirmation = data['volume_trend'] * data['vol_adj_volume_accel']
    data['volume_confirmed_divergence'] = data['acceleration_divergence'] * volume_confirmation
    
    # Scale by current day's volume relative to volatility-adjusted average
    vol_adj_volume_avg = (data['volume'] / (data['volatility_20d'] + 1e-8)).rolling(window=20).mean()
    data['volume_scaling'] = (data['volume'] / (data['volatility_20d'] + 1e-8)) / vol_adj_volume_avg
    
    # Regime-Adaptive Signal Enhancement
    # Price pressure calculation
    data['price_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Volatility-adjusted price pressure thresholds
    vol_adj_pressure_threshold = np.where(
        data['vol_regime'] == 2, 0.6,  # Higher threshold in high volatility
        np.where(data['vol_regime'] == 0, 0.4, 0.5)  # Lower threshold in low volatility
    )
    
    # Final factor calculation
    base_factor = data['volume_confirmed_divergence'] * data['volume_scaling']
    
    # Apply price pressure filter
    pressure_filter = np.where(
        (base_factor > 0) & (data['price_pressure'] > vol_adj_pressure_threshold), 1,
        np.where(
            (base_factor < 0) & (data['price_pressure'] < (1 - vol_adj_pressure_threshold)), 1, 0.5
        )
    )
    
    data['factor'] = base_factor * pressure_filter
    
    # Adjust sensitivity based on volatility regime and momentum direction
    regime_sensitivity = np.where(
        (data['vol_regime'] == 2) & (data['momentum_5d'] > 0), 0.7,  # Reduce sensitivity in high vol up moves
        np.where(
            (data['vol_regime'] == 0) & (data['momentum_5d'] < 0), 1.3,  # Increase sensitivity in low vol down moves
            1.0
        )
    )
    
    data['final_factor'] = data['factor'] * regime_sensitivity
    
    return data['final_factor']
