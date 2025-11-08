import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Hierarchy with Volatility Regime and Multi-Timeframe Trend Convergence
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Price-Volume Divergence Hierarchy
    # Short-term (5-day) components
    data['price_return_5d'] = data['close'].pct_change(5)
    data['price_slope_5d'] = data['close'].rolling(5).apply(lambda x: (x[-1] - x[0]) / 4 if len(x) == 5 else np.nan)
    
    data['volume_change_5d'] = data['volume'].pct_change(5)
    data['volume_slope_5d'] = data['volume'].rolling(5).apply(lambda x: (x[-1] - x[0]) / 4 if len(x) == 5 else np.nan)
    
    # Short-term divergence score
    data['short_divergence'] = np.where(
        (data['price_return_5d'] > 0) & (data['volume_change_5d'] < 0),
        -data['price_slope_5d'] * abs(data['volume_slope_5d']),
        np.where(
            (data['price_return_5d'] < 0) & (data['volume_change_5d'] > 0),
            data['price_slope_5d'] * abs(data['volume_slope_5d']),
            0
        )
    )
    
    # Medium-term (20-day) components
    data['price_return_20d'] = data['close'].pct_change(20)
    data['ma_20'] = data['close'].rolling(20).mean()
    data['price_slope_20d'] = data['ma_20'].diff(5) / 5
    
    data['volume_ma_20'] = data['volume'].rolling(20).mean()
    data['volume_slope_20d'] = data['volume_ma_20'].diff(5) / 5
    
    # Medium-term divergence persistence
    data['medium_divergence'] = np.where(
        (data['price_return_20d'] > 0) & (data['volume_slope_20d'] < 0),
        -data['price_slope_20d'] * abs(data['volume_slope_20d']),
        np.where(
            (data['price_return_20d'] < 0) & (data['volume_slope_20d'] > 0),
            data['price_slope_20d'] * abs(data['volume_slope_20d']),
            0
        )
    )
    
    # Volatility Regime Framework
    data['volatility_10d'] = data['close'].pct_change().rolling(10).std()
    data['volatility_50d'] = data['close'].pct_change().rolling(50).std()
    data['volatility_ratio'] = data['volatility_10d'] / data['volatility_50d']
    
    # Regime classification
    data['vol_regime'] = np.where(
        data['volatility_ratio'] > 1.5, 2,  # High volatility
        np.where(data['volatility_ratio'] < 0.7, 0, 1)  # Low volatility, Normal volatility
    )
    
    # Multi-Timeframe Trend Convergence
    # Ultra-short trend (3-day)
    data['trend_3d'] = data['close'].pct_change(3)
    data['volume_trend_3d'] = data['volume'].pct_change(3)
    
    # Short-term trend (10-day)
    data['ma_10'] = data['close'].rolling(10).mean()
    data['trend_10d'] = (data['close'] - data['ma_10']) / data['ma_10']
    data['volume_trend_10d'] = data['volume'].pct_change(10)
    
    # Medium-term trend (30-day)
    data['ma_30'] = data['close'].rolling(30).mean()
    data['trend_30d'] = (data['close'] - data['ma_30']) / data['ma_30']
    data['volume_trend_30d'] = data['volume'].pct_change(30)
    
    # Trend convergence scoring
    trend_signs = np.sign(data[['trend_3d', 'trend_10d', 'trend_30d']])
    data['trend_alignment'] = trend_signs.sum(axis=1) / 3
    
    # Volume-weighted convergence
    volume_weights = abs(data[['volume_trend_3d', 'volume_trend_10d', 'volume_trend_30d']])
    volume_weights = volume_weights.div(volume_weights.sum(axis=1), axis=0)
    data['volume_weighted_trend'] = (trend_signs * volume_weights).sum(axis=1)
    
    # Composite Alpha Integration
    # Dynamic factor weighting by volatility regime
    regime_weights = {
        0: [0.4, 0.3, 0.3],  # Low volatility: emphasis on divergence
        1: [0.3, 0.4, 0.3],  # Normal volatility: balanced
        2: [0.2, 0.3, 0.5]   # High volatility: emphasis on trend convergence
    }
    
    # Normalize divergence signals
    short_div_norm = data['short_divergence'] / data['short_divergence'].rolling(50).std()
    medium_div_norm = data['medium_divergence'] / data['medium_divergence'].rolling(50).std()
    
    # Combined divergence factor
    divergence_factor = (short_div_norm + medium_div_norm) / 2
    
    # Final composite factor
    composite_factor = pd.Series(index=data.index, dtype=float)
    
    for regime in [0, 1, 2]:
        mask = data['vol_regime'] == regime
        weights = regime_weights[regime]
        composite_factor[mask] = (
            weights[0] * divergence_factor[mask] +
            weights[1] * data.loc[mask, 'trend_alignment'] +
            weights[2] * data.loc[mask, 'volume_weighted_trend']
        )
    
    # Fill any remaining NaN values
    composite_factor = composite_factor.fillna(0)
    
    return composite_factor
